
def func1(x,y,**kwargs):
    print(x,y)
    print(kwargs,"kwargs")

    assert "a" in kwargs

    a = kwargs.pop("a")

def func2():
    x = 1
    y = 2.0
    kwargs0 = dict(
        a="1a",
        b="1b"
    )
    for i in range(10):
        x += i
        func1(
            x,
            y,
            **kwargs0
        )


func2()