# 💻 20875 - Software Engineering
To host the code used for assignments, homework and anything else related to the course "20875 - Software Engineering" I took during the first semester of my MSc in Artificial Intelligence at Bocconi University in Milan, Italy.  

## Assignment 01 - Boolean Compiler
The directory `./HW01/` contains the files related to the first assignment of the course which required to develop a compiler to parse a raw `input.txt` encoding instructions to solve boolean expressions and display truth tables.  
I tackled this problem by creating a unique `Compiler` class contained in `table.py` which takes care of all the processing. First, it tokenizes the raw text into separate instructions containing keywords, variables names and boolean operators. Then it solves each boolean expression via recursion. Upon request, a truth table can be shown.

# 💻 20874 - Algorithms for Optimization and Inference
To host the code used for assignments, homework and anything else related to the course "20875 - Algorithms for Optimization and Inference" I took during the first semester of my MSc in Artificial Intelligence at Bocconi University in Milan, Italy.  
- `.mod` scripts used to solve linear programming (LP) problems via *glpk*
- `.py`  scripts to integrate with *glpk* for more complex problems

## Assignemnt 01 - Image Compression via LP
Develop an algorithm to compress RGB images using linear programming. Then use *glpk* to run the algorithm and compress a sample image.