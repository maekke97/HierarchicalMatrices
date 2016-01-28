# Hierarchical Matrices

## Project Proposal

### Project Layout

1. Implement the framework for hierarchical matrices in python
1. Test the code for correctness and performance on a known small problem (compare against existing C-library)
1. Run a "real-life" large-scale simulation with it
1. Write a summary of the theory, implementation and simulation

### Details

#### Implementation

1. Use the algorithms from the lecture script / Hackbusch plus the implemented Matlab code  
Python suits well because of object oriented structure of the problem  
**Possible issues:**
    - Check for fast linear algebra libraries
1. Maybe build a complete package to include in SciPy project
1. Maybe implement a parallel version

#### Tests

1. Use the small examples from the lecture (Gallerkin)
1. Run tests for correctness
1. Run tests for performance and perform the same tests with the C-library

#### Simulation

1. Find a problem that is suitable for h-matrix approximation
1. Implement the problem set-up
1. Run the simulation

#### Written Summary

1. Summarize the theory
1. Description of implementation
    1. General description of algorithms
    1. Problems / traps in the implementation and their solutions
    1. Complete analysis of resource usage
1. Short summary of performance tests
1. Description of the physical problem and the simulated results