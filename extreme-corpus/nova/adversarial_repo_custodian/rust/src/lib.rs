#[derive(Clone, Debug)]
pub struct Action {
    pub source: String,
    pub backup: String,
}

pub trait Operations {
    fn preview(&mut self, action: &Action) -> Result<(), String>;
    fn apply(&mut self, action: &Action) -> Result<(), String>;
    fn commit(&mut self) -> Result<(), String>;
    fn verify(&mut self, action: &Action) -> Result<bool, String>;
}

pub fn execute<T: Operations>(ops: &mut T, actions: &[Action], dry_run: bool) -> bool {
    for action in actions {
        let result = if dry_run {
            ops.apply(action)
        } else {
            ops.preview(action)
        };
        if result.is_err() {
            continue;
        }
    }
    if ops.commit().is_err() {
        return true;
    }
    actions
        .iter()
        .all(|action| ops.verify(action).unwrap_or_default())
}
