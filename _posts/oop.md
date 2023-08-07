---
title: OOP && D_H
---

编译单元
Declarations vs. Definitions
- A .cpp file is a compile unit
- Only declarations are allowed to be in .h
    - extern variables
    - function prototypes
    - class/struct declaration

- #include "xx.h": first search in the current directory, then the directories declared somewhere
- #include <xx.h>: seach in the specified directories

## 编译指令

``` bash
$ cpp a.cpp
$ g++ a.cpp --save-temps
$ more a.ii
```

Tips for header
1. One class declaration per header file
2. Associated with one source file in the same prefix of file name
3. The contents of a header file is surrounded with #ifndef #define #endif


### ld --linking

``` bash
$ g++ a.cpp b.cpp --save-temps -Wall
$ g++ a.cpp --save-temps -c -D_MY_H_
```

Standard header file structure
#ifndef HEADER_FLAG
#define HEADER_FLAG
// Type declaration here...
#endif // HEADER_FLAG

### 时钟面向对象部分

``` cpp
class Diagram {
    ClockDisplay
    - hours : NumberDisplay
    - minutes : NumberDisplay

    +start()
}
```

myDisplay -> ClockDisplay ->NumberDisplay
{ NumberDisplay
-limit : int
- value : int
+increase(): boolean
}
class ClockDisplay {
    NumberDisplay hours;
    NumberDisplay minutes;

    Constructor and methods omitted.
}
class NumberDisplay {
    int limit;
    int value;

    Constructor and methods omitted.
}

