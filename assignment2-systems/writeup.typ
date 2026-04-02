#set document(
    title: [Writeup for Assignment2-Systems],
)

#set heading(numbering: "1.1.a.")
#set text(
    font: "New Computer Modern",
)

#show title: set align(center)
#show outline.entry.where(level: 1): set text(weight: "bold")
#title()
#outline()
= Assignment Overview
The assignment works with model configurations as follows:
#show table.cell.where(y: 0): strong
#set table(
    stroke: (x, y) => (
        if y == 0 { (top: 1pt + black, bottom: 1pt + black) } else if y == 5 { (bottom: 1pt + black) }
    ),
    align: (x, y) => (
        if x > 0 { center } else { left }
    ),
)
#align(center)[
    #table(
        columns: 5,
        table.header([Size], [d_mdoel], [d_ff], [num_layers], [num_heads]),
        [small], [768], [3072], [12], [12],
        [medium], [1024], [4096], [24], [16],
        [large], [1280], [5120], [36], [20],
        [xl], [1600], [6400], [48], [25],
        [2.7B], [2560], [10240], [32], [32],
    )
]
== Problem (benchmarking\_script) Write The Basic Profiling Infrastructure.

=== Write a script to perform basic end-to-end benchmarking of the forward and backward passes in your model. 
Specifically, your script should support the following:
- Given hyperparameters (e.g., number of layers), initialize a model.
- Generate a random batch of data.
- Run w warm-up steps
In the data we use `torch.long` instead of `torch.uint64`. That is, in deep learning frameworks like PyTorch and most systems programming, signed integers (`int64` or `Long`) are used for indexing because they are actually safer and often faster than unsigned integers.

It's implemented as `cs336_systems/benchmark.py`
