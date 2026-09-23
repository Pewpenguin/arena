use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

pub(super) struct MockServer {
    pub origin: String,
    pub request: oneshot::Receiver<Vec<u8>>,
    pub handle: tokio::task::JoinHandle<()>,
}

pub(super) async fn start_mock(status: u16, reason: &str, body: &str) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel();
    let reason = reason.to_string();
    let body = body.to_string();

    let handle = tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };

        let request = read_http_request(&mut stream).await;
        let _ = tx.send(request);
        let content_length = body.len();
        let headers = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(headers.as_bytes()).await;
        let _ = stream.write_all(body.as_bytes()).await;
        let _ = stream.shutdown().await;
    });

    MockServer {
        origin: format!("http://{addr}"),
        request: rx,
        handle,
    }
}

async fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut data = Vec::new();
    loop {
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).await.expect("read request");
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        if let Some(headers_end) = find_headers_end(&data) {
            let content_length = content_length(&data[..headers_end]).unwrap_or(0);
            if data.len() >= headers_end + content_length {
                data.truncate(headers_end + content_length);
                break;
            }
        }
    }
    data
}

fn find_headers_end(data: &[u8]) -> Option<usize> {
    data.windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

fn content_length(headers: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(headers).ok()?;
    for line in text.split("\r\n") {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            return value.trim().parse().ok();
        }
    }
    None
}

pub(super) fn header_value(request: &str, name: &str) -> Option<String> {
    request.lines().find_map(|line| {
        let (header_name, value) = line.split_once(':')?;
        header_name
            .eq_ignore_ascii_case(name)
            .then_some(value.trim().to_string())
    })
}

pub(super) fn request_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("request path")
}

pub(super) fn request_body(raw: &[u8]) -> serde_json::Value {
    let headers_end = find_headers_end(raw).expect("headers");
    serde_json::from_slice(&raw[headers_end..]).expect("json body")
}

pub(super) struct CrossOriginRedirect {
    pub origin: String,
    pub saw_secret: Arc<AtomicBool>,
    source: tokio::task::JoinHandle<()>,
    capture: tokio::task::JoinHandle<()>,
}

impl CrossOriginRedirect {
    pub(super) fn abort(self) {
        self.source.abort();
        self.capture.abort();
    }
}

pub(super) async fn start_cross_origin_redirect(secret: &str) -> CrossOriginRedirect {
    let saw_secret = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&saw_secret);
    let secret = secret.to_string();

    let capture_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind capture");
    let capture_addr = capture_listener.local_addr().expect("capture addr");
    let capture = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = capture_listener.accept().await else {
                return;
            };
            let request = read_http_request(&mut stream).await;
            if String::from_utf8_lossy(&request).contains(&secret) {
                flag.store(true, Ordering::SeqCst);
            }
            let body = "ok";
            let content_length = body.len();
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(headers.as_bytes()).await;
            let _ = stream.write_all(body.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });

    let source_listener = TcpListener::bind("127.0.0.1:0").await.expect("bind source");
    let source_addr = source_listener.local_addr().expect("source addr");
    let location = format!("http://{capture_addr}/stolen");
    let source = tokio::spawn(async move {
        let Ok((mut stream, _)) = source_listener.accept().await else {
            return;
        };
        let _ = read_http_request(&mut stream).await;
        let headers = format!(
            "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(headers.as_bytes()).await;
        let _ = stream.shutdown().await;
    });

    CrossOriginRedirect {
        origin: format!("http://{source_addr}"),
        saw_secret,
        source,
        capture,
    }
}
