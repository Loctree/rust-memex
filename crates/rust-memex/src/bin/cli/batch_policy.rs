use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct BatchFailurePolicy {
    pub strict: bool,
    pub max_failure_rate: f64,
}

impl BatchFailurePolicy {
    pub fn new(strict: bool, max_failure_rate: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&max_failure_rate) {
            bail!("--max-failure-rate must be between 0.0 and 1.0");
        }
        Ok(Self {
            strict,
            max_failure_rate,
        })
    }

    pub fn should_fail(&self, summary: &BatchRunSummary) -> bool {
        (self.strict && summary.failed > 0) || summary.failure_rate() > self.max_failure_rate
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchRunSummary {
    pub indexed: usize,
    pub failed: usize,
    pub total: usize,
    pub failure_rate: f64,
    pub errors: Vec<String>,
}

impl BatchRunSummary {
    pub fn new(indexed: usize, failed: usize, total: usize, errors: Vec<String>) -> Self {
        let failure_rate = if total == 0 {
            0.0
        } else {
            failed as f64 / total as f64
        };
        Self {
            indexed,
            failed,
            total,
            failure_rate,
            errors,
        }
    }

    fn failure_rate(&self) -> f64 {
        self.failure_rate
    }

    pub fn emit_json(&self) -> Result<()> {
        println!("{}", serde_json::to_string(self)?);
        Ok(())
    }

    pub fn emit_warning(&self, noun: &str) {
        if self.failed > 0 {
            eprintln!(
                "WARNING: {}/{} {} failed. See log above.",
                self.failed, self.total, noun
            );
        }
    }

    pub fn enforce(&self, policy: BatchFailurePolicy, noun: &str) -> Result<()> {
        if policy.should_fail(self) {
            let reason = if policy.strict && self.failed > 0 {
                format!("strict=true and {} failure(s) were observed", self.failed)
            } else {
                format!(
                    "failure_rate {:.4} > max_failure_rate {:.4}",
                    self.failure_rate, policy.max_failure_rate
                )
            };
            bail!("{}/{} {} failed ({reason})", self.failed, self.total, noun);
        }
        Ok(())
    }
}
