# Programming structure of project **A**lternately **U**pdating **M**otion and **I**mage **SART**

## Components
1. Projection operations
    - [x] .cu codes
    - [x] .h codes.
    - [ ] test program 
2. Backprojection operations
    - [x] .cu codes
    - [x] .h codes.
    - [ ] test program 
3. Mathematics operations
    - [x] Add 
    - [x] Division
    - [x] Initial 
4. Deform operations
    - [x] .cu codes
    - [x] .h codes.
    - [ ] test program 
5. UDVF updating operations
    - [x] .cu codes
    - [x] .h codes.
    - [ ] test program 
6. Stopping criteria
    - [ ] stopping criteria for motion model updating
    - [ ] stopping criteria for image updating



## Implementation process
1. Use regular _SART_ to obtain blurred image from no-motion-model SART with components 1,2,3 only, for _n1_ times.
2. Update motion model, by iteratively applying projection process on every angle, for __n2__ times max. set a stop criteria
3. Update image, with estimated motion model, for __n3__ times maximum. set a stop criteria.

**Note we need to moniter the error on projection.**
 





