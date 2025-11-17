# Savi2025-2026-assignment1-group9
Task 1
---

This script demonstrates a complete workflow for point cloud alignment using the Iterative Closest Point (ICP) algorithm in Open3D. It begins by loading a pair of sample point clouds and applying an initial transformation to provide a rough alignment. 
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/689127c2-94f3-4ea2-a246-4f330bef1958" />

The code then evaluates this initial guess before performing *point-to-plane ICP*, an optimization-based method that iteratively refines the transformation by minimizing geometric distance between corresponding points. The result is an improved alignment along with the final estimated transformation matrix, which is visualized before and after refinement. 

---
