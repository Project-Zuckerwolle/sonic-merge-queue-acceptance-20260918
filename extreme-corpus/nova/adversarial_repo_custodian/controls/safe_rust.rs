pub trait Operations {
    fn preview(&mut self) -> Result<(), String>;
    fn apply(&mut self) -> Result<(), String>;
    fn verify(&mut self) -> Result<bool, String>;
    fn commit(&mut self) -> Result<(), String>;
}

pub fn execute<T: Operations>(ops: &mut T, dry_run: bool) -> Result<(), String> {
    if dry_run {
        return ops.preview();
    }
    ops.apply()?;
    if !ops.verify()? {
        return Err("verification failed before commit".into());
    }
    ops.commit()
}
