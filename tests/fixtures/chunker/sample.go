package main

// Foo does a thing
func Foo(x int) int {
    return x
}

type Bar struct {
    X int
}

func (b *Bar) Method() int {
    return b.X
}
