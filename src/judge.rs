use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::evaluate::EvaluatedResult;
use crate::provider::{
    CompletionRequest, ModelId, ModelProvider, PROVIDER_CONCURRENCY, ProviderError,
};
use crate::task::Task;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgeDecision {
    A,
    B,
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Judgment {
    pub task_id: String,
    pub model_a: ModelId,
    pub model_b: ModelId,
    pub judge_model: ModelId,
    pub winner: JudgeDecision,
    pub reason: String,
    pub duration_ms: u64,
    pub agreement: bool,
    /// Original presentation `(model_a, model_b)`, in that identity frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation_ab: Option<JudgeDecision>,
    /// Swapped presentation `(model_b, model_a)`, mapped back to the `(model_a, model_b)` frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation_ba: Option<JudgeDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDecision {
    pub winner: JudgeDecision,
    pub reason: String,
}

#[derive(Debug, Error)]
pub enum JudgeError {
    #[error("judge failed for task {task_id} with judge model {judge_model}: {source}")]
    Provider {
        task_id: String,
        judge_model: ModelId,
        #[source]
        source: ProviderError,
    },
    #[error("invalid judge response JSON: {0}")]
    InvalidJson(String),
    #[error(
        "invalid judge response for task {task_id} with models {model_a} vs {model_b} (judge {judge_model}): {source}"
    )]
    InvalidResponse {
        task_id: String,
        model_a: ModelId,
        model_b: ModelId,
        judge_model: ModelId,
        #[source]
        source: Box<JudgeError>,
    },
    #[error("cannot judge results from different tasks")]
    DifferentTasks,
}

pub fn build_judge_prompt(task: &Task, response_a: &str, response_b: &str) -> String {
    format!(
        "You are comparing two model responses to the same task.\n\
         \n\
         Task:\n\
         {prompt}\n\
         \n\
         Response A:\n\
         {response_a}\n\
         \n\
         Response B:\n\
         {response_b}\n\
         \n\
         Which response better satisfies the user's task?\n\
         \n\
         Reply with only valid JSON in exactly one of these forms:\n\
         {{\"winner\":\"a\",\"reason\":\"<short explanation>\"}}\n\
         {{\"winner\":\"b\",\"reason\":\"<short explanation>\"}}\n\
         {{\"winner\":\"draw\",\"reason\":\"<short explanation>\"}}",
        prompt = task.prompt,
        response_a = response_a,
        response_b = response_b,
    )
}

pub fn parse_decision(text: &str) -> Result<ParsedDecision, JudgeError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Payload {
        winner: JudgeDecision,
        reason: String,
    }

    let trimmed = text.trim();
    let mut search_from = 0;
    let mut found = None;

    while let Some(offset) = trimmed[search_from..].find('{') {
        let start = search_from + offset;
        let mut deserializer = serde_json::Deserializer::from_str(&trimmed[start..]);
        if let Ok(payload) = Payload::deserialize(&mut deserializer) {
            found = Some(ParsedDecision {
                winner: payload.winner,
                reason: payload.reason,
            });
        }
        search_from = start + 1;
    }

    found.ok_or_else(|| {
        let preview: String = trimmed.chars().take(500).collect();
        JudgeError::InvalidJson(format!("no valid judgment JSON found; response: {preview}"))
    })
}

fn map_swapped_winner(winner: JudgeDecision) -> JudgeDecision {
    match winner {
        JudgeDecision::A => JudgeDecision::B,
        JudgeDecision::B => JudgeDecision::A,
        JudgeDecision::Draw => JudgeDecision::Draw,
    }
}

fn resolve_winners(original: JudgeDecision, swapped: JudgeDecision) -> (JudgeDecision, bool) {
    let mapped = map_swapped_winner(swapped);
    if original == mapped {
        (original, true)
    } else {
        (JudgeDecision::Draw, false)
    }
}

fn resolve_judgments(original: Judgment, swapped: Judgment, duration_ms: u64) -> Judgment {
    let (winner, agreement) = resolve_winners(original.winner.clone(), swapped.winner.clone());
    let reason = if agreement {
        original.reason
    } else {
        format!(
            "position bias disagreement (original: {}; swapped: {})",
            original.reason, swapped.reason
        )
    };

    Judgment {
        task_id: original.task_id,
        model_a: original.model_a,
        model_b: original.model_b,
        judge_model: original.judge_model,
        winner,
        reason,
        duration_ms,
        agreement,
        orientation_ab: Some(original.winner),
        orientation_ba: Some(map_swapped_winner(swapped.winner)),
    }
}

pub async fn judge_pair(
    provider: &impl ModelProvider,
    judge_model: ModelId,
    task: &Task,
    result_a: &EvaluatedResult,
    result_b: &EvaluatedResult,
) -> Result<Judgment, JudgeError> {
    if result_a.task_id != result_b.task_id {
        return Err(JudgeError::DifferentTasks);
    }

    let prompt = build_judge_prompt(task, &result_a.response.text, &result_b.response.text);
    let request = CompletionRequest {
        model: judge_model.clone(),
        prompt,
    };
    let started = Instant::now();
    let response = provider
        .complete(request)
        .await
        .map_err(|source| JudgeError::Provider {
            task_id: task.id.clone(),
            judge_model: judge_model.clone(),
            source,
        })?;
    let duration_ms = started.elapsed().as_millis() as u64;

    let decision =
        parse_decision(&response.text).map_err(|source| JudgeError::InvalidResponse {
            task_id: task.id.clone(),
            model_a: result_a.model.clone(),
            model_b: result_b.model.clone(),
            judge_model: judge_model.clone(),
            source: Box::new(source),
        })?;

    Ok(Judgment {
        task_id: result_a.task_id.clone(),
        model_a: result_a.model.clone(),
        model_b: result_b.model.clone(),
        judge_model,
        winner: decision.winner,
        reason: decision.reason,
        duration_ms,
        agreement: true,
        orientation_ab: None,
        orientation_ba: None,
    })
}

pub async fn judge_pairs<P>(
    provider: &P,
    judge_model: ModelId,
    task: &Task,
    results: &[EvaluatedResult],
    mut on_complete: impl FnMut(&Judgment),
) -> Result<Vec<Judgment>, JudgeError>
where
    P: ModelProvider + Clone + Send + 'static,
{
    let pair_indices: Vec<_> = (0..results.len())
        .flat_map(|i| ((i + 1)..results.len()).map(move |j| (i, j)))
        .collect();
    let pair_count = pair_indices.len();

    let semaphore = Arc::new(Semaphore::new(PROVIDER_CONCURRENCY));
    let mut set = JoinSet::new();

    for (index, (i, j)) in pair_indices.into_iter().enumerate() {
        for (orientation, (result_a, result_b)) in [
            (0u8, (results[i].clone(), results[j].clone())),
            (1u8, (results[j].clone(), results[i].clone())),
        ] {
            let provider = provider.clone();
            let judge_model = judge_model.clone();
            let task = task.clone();
            let semaphore = semaphore.clone();
            set.spawn(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .expect("provider semaphore is not closed");
                let started = Instant::now();
                (
                    index,
                    orientation,
                    started,
                    judge_pair(&provider, judge_model, &task, &result_a, &result_b).await,
                )
            });
        }
    }

    let mut original = vec![None; pair_count];
    let mut swapped = vec![None; pair_count];
    let mut ordered = vec![None; pair_count];

    while let Some(joined) = set.join_next().await {
        let (index, orientation, started, result) = joined.expect("pairwise judging panicked");
        let judgment = result?;
        on_complete(&judgment);

        match orientation {
            0 => original[index] = Some((started, judgment)),
            _ => swapped[index] = Some((started, judgment)),
        }

        if original[index].is_some() && swapped[index].is_some() {
            let (orig_started, orig) = original[index].take().expect("original judgment");
            let (swap_started, swap) = swapped[index].take().expect("swapped judgment");
            let duration_ms = orig_started.min(swap_started).elapsed().as_millis() as u64;
            ordered[index] = Some(resolve_judgments(orig, swap, duration_ms));
        }
    }

    Ok(ordered
        .into_iter()
        .map(|result| result.expect("missing judgment"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tokio::sync::Notify;

    use crate::provider::{CompletionResponse, PROVIDER_CONCURRENCY};

    #[test]
    fn parses_winner_a() {
        let decision = parse_decision(r#"{"winner":"a","reason":"A is better"}"#).unwrap();
        assert_eq!(decision.winner, JudgeDecision::A);
        assert_eq!(decision.reason, "A is better");
    }

    #[test]
    fn parses_winner_b() {
        let decision = parse_decision(r#"{"winner":"b","reason":"B is better"}"#).unwrap();
        assert_eq!(decision.winner, JudgeDecision::B);
    }

    #[test]
    fn parses_draw() {
        let decision = parse_decision(r#"{"winner":"draw","reason":"equal"}"#).unwrap();
        assert_eq!(decision.winner, JudgeDecision::Draw);
    }

    #[test]
    fn chooses_last_valid_judgment_object() {
        let text = r#"{"winner":"a","reason":"echoed"}
{"winner":"b","reason":"final"}"#;
        let decision = parse_decision(text).unwrap();
        assert_eq!(decision.winner, JudgeDecision::B);
        assert_eq!(decision.reason, "final");
    }

    #[test]
    fn parses_json_embedded_in_surrounding_text() {
        let text =
            "<think>\nreasoning about the answers\n</think>\n{\"winner\":\"b\",\"reason\":\"B\"}\n";
        let decision = parse_decision(text).unwrap();
        assert_eq!(decision.winner, JudgeDecision::B);
        assert_eq!(decision.reason, "B");
    }

    #[test]
    fn rejects_invalid_winner() {
        let error = parse_decision(r#"{"winner":"c","reason":"no"}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_missing_winner() {
        let error = parse_decision(r#"{"reason":"no winner"}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_missing_reason() {
        let error = parse_decision(r#"{"winner":"a"}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_non_string_reason() {
        let error = parse_decision(r#"{"winner":"a","reason":1}"#).unwrap_err();
        assert!(matches!(error, JudgeError::InvalidJson(_)));
    }

    #[test]
    fn rejects_response_without_judgment_json() {
        let error = parse_decision("<think>no judgment object here</think>").unwrap_err();
        match error {
            JudgeError::InvalidJson(message) => {
                assert!(message.contains("no valid judgment JSON found"));
                assert!(message.contains("response:"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn resolves_agreement_and_disagreement() {
        assert_eq!(
            resolve_winners(JudgeDecision::A, JudgeDecision::B),
            (JudgeDecision::A, true)
        );
        assert_eq!(
            resolve_winners(JudgeDecision::B, JudgeDecision::A),
            (JudgeDecision::B, true)
        );
        assert_eq!(
            resolve_winners(JudgeDecision::Draw, JudgeDecision::Draw),
            (JudgeDecision::Draw, true)
        );
        assert_eq!(
            resolve_winners(JudgeDecision::A, JudgeDecision::A),
            (JudgeDecision::Draw, false)
        );
        assert_eq!(
            resolve_winners(JudgeDecision::Draw, JudgeDecision::A),
            (JudgeDecision::Draw, false)
        );
    }

    fn evaluated(model: &str, text: &str) -> EvaluatedResult {
        EvaluatedResult {
            task_id: "t1".into(),
            model: ModelId::new(model),
            response: CompletionResponse { text: text.into() },
            evaluation: None,
            duration_ms: 0,
        }
    }

    #[derive(Clone)]
    struct AlwaysPositionA;

    impl ModelProvider for AlwaysPositionA {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(CompletionResponse {
                text: r#"{"winner":"a","reason":"position a"}"#.into(),
            })
        }
    }

    #[tokio::test]
    async fn position_always_a_resolves_to_draw() {
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let results = vec![evaluated("m0", "left"), evaluated("m1", "right")];

        let judgments = judge_pairs(
            &AlwaysPositionA,
            ModelId::new("judge"),
            &task,
            &results,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(judgments.len(), 1);
        assert_eq!(judgments[0].model_a, ModelId::new("m0"));
        assert_eq!(judgments[0].model_b, ModelId::new("m1"));
        assert_eq!(judgments[0].winner, JudgeDecision::Draw);
        assert!(!judgments[0].agreement);
        assert_eq!(judgments[0].orientation_ab, Some(JudgeDecision::A));
        assert_eq!(judgments[0].orientation_ba, Some(JudgeDecision::B));
        assert!(judgments[0].reason.contains("position bias disagreement"));
    }

    #[derive(Clone)]
    struct InvalidJsonJudge;

    impl ModelProvider for InvalidJsonJudge {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(CompletionResponse {
                text: "not a judgment".into(),
            })
        }
    }

    #[tokio::test]
    async fn malformed_judge_response_identifies_task_and_pair() {
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let error = judge_pair(
            &InvalidJsonJudge,
            ModelId::new("judge"),
            &task,
            &evaluated("m0", "left"),
            &evaluated("m1", "right"),
        )
        .await
        .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("t1"), "{message}");
        assert!(message.contains("m0"), "{message}");
        assert!(message.contains("m1"), "{message}");
        assert!(message.contains("judge"), "{message}");

        match error {
            JudgeError::InvalidResponse {
                task_id,
                model_a,
                model_b,
                judge_model,
                source,
            } => {
                assert_eq!(task_id, "t1");
                assert_eq!(model_a, ModelId::new("m0"));
                assert_eq!(model_b, ModelId::new("m1"));
                assert_eq!(judge_model, ModelId::new("judge"));
                assert!(matches!(*source, JudgeError::InvalidJson(_)));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[derive(Clone)]
    struct PrefersBetterText;

    impl ModelProvider for PrefersBetterText {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let a_is_better = request.prompt.contains("Response A:\nbetter\n");
            let winner = if a_is_better { "a" } else { "b" };
            Ok(CompletionResponse {
                text: format!(r#"{{"winner":"{winner}","reason":"prefers better"}}"#),
            })
        }
    }

    #[tokio::test]
    async fn consistent_preference_survives_position_swap() {
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let results = vec![evaluated("m0", "better"), evaluated("m1", "worse")];

        let judgments = judge_pairs(
            &PrefersBetterText,
            ModelId::new("judge"),
            &task,
            &results,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(judgments.len(), 1);
        assert_eq!(judgments[0].winner, JudgeDecision::A);
        assert!(judgments[0].agreement);
        assert_eq!(judgments[0].orientation_ab, Some(JudgeDecision::A));
        assert_eq!(judgments[0].orientation_ba, Some(JudgeDecision::A));
        assert_eq!(judgments[0].reason, "prefers better");
    }

    #[derive(Clone)]
    struct GateJudgeProvider {
        released: Arc<Mutex<bool>>,
        notify: Arc<Notify>,
    }

    impl ModelProvider for GateJudgeProvider {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let involves_slow = request.prompt.contains("\nslow\n");
            let involves_fast =
                request.prompt.contains("\nfast\n") && request.prompt.contains("\nm0\n");

            if involves_slow {
                loop {
                    let notified = self.notify.notified();
                    if *self.released.lock().expect("released") {
                        break;
                    }
                    notified.await;
                }
            } else if involves_fast {
                *self.released.lock().expect("released") = true;
                self.notify.notify_waiters();
            }

            Ok(CompletionResponse {
                text: r#"{"winner":"a","reason":"ok"}"#.into(),
            })
        }
    }

    #[tokio::test]
    async fn judgments_follow_pair_order_when_completion_order_differs() {
        let provider = GateJudgeProvider {
            released: Arc::new(Mutex::new(false)),
            notify: Arc::new(Notify::new()),
        };
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let results = vec![
            evaluated("m0", "m0"),
            evaluated("m1", "slow"),
            evaluated("m2", "fast"),
        ];

        let judgments = judge_pairs(&provider, ModelId::new("judge"), &task, &results, |_| {})
            .await
            .unwrap();

        assert_eq!(judgments.len(), 3);
        assert_eq!(judgments[0].model_a, ModelId::new("m0"));
        assert_eq!(judgments[0].model_b, ModelId::new("m1"));
        assert_eq!(judgments[1].model_a, ModelId::new("m0"));
        assert_eq!(judgments[1].model_b, ModelId::new("m2"));
        assert_eq!(judgments[2].model_a, ModelId::new("m1"));
        assert_eq!(judgments[2].model_b, ModelId::new("m2"));
    }

    #[derive(Clone)]
    struct HoldFirstWave {
        entered: Arc<AtomicUsize>,
        released: Arc<Mutex<bool>>,
        notify: Arc<Notify>,
    }

    impl ModelProvider for HoldFirstWave {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let n = self.entered.fetch_add(1, Ordering::SeqCst);
            if n < PROVIDER_CONCURRENCY {
                loop {
                    let notified = self.notify.notified();
                    if *self.released.lock().expect("released") {
                        break;
                    }
                    notified.await;
                }
            }

            Ok(CompletionResponse {
                text: r#"{"winner":"a","reason":"ok"}"#.into(),
            })
        }
    }

    #[tokio::test]
    async fn pair_duration_excludes_provider_semaphore_wait() {
        let entered = Arc::new(AtomicUsize::new(0));
        let released = Arc::new(Mutex::new(false));
        let notify = Arc::new(Notify::new());
        let provider = HoldFirstWave {
            entered: entered.clone(),
            released: released.clone(),
            notify: notify.clone(),
        };
        let task = Task {
            id: "t1".into(),
            prompt: "p".into(),
            evaluation: None,
        };
        let results = vec![
            evaluated("m0", "m0"),
            evaluated("m1", "m1"),
            evaluated("m2", "m2"),
            evaluated("m3", "m3"),
        ];

        let (judgments, ()) = tokio::join!(
            judge_pairs(&provider, ModelId::new("judge"), &task, &results, |_| {}),
            async {
                while entered.load(Ordering::SeqCst) < PROVIDER_CONCURRENCY {
                    tokio::task::yield_now().await;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                *released.lock().expect("released") = true;
                notify.notify_waiters();
            }
        );
        let judgments = judgments.unwrap();
        let min = judgments.iter().map(|j| j.duration_ms).min().unwrap();
        let max = judgments.iter().map(|j| j.duration_ms).max().unwrap();

        assert!(
            min < max,
            "queued pair duration {min}ms should exclude the semaphore wait included in occupying duration {max}ms"
        );
    }
}
