// leading comment
fn foo(x: i32) -> i32 {
    x + 1
}

#[derive(Debug)]
struct Bar {
    x: i32,
}

impl Bar {
    fn method(&self) -> i32 {
        self.x
    }
}
