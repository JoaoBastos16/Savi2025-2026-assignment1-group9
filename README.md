# Savi2025-2026-assignment1-group9
Task 1
---

This script demonstrates a complete workflow for point cloud alignment using the Iterative Closest Point (ICP) algorithm in Open3D. It begins by loading a pair of sample point clouds and applying an initial transformation to provide a rough alignment. The code then evaluates this initial guess before performing *point-to-plane ICP*, an optimization-based method that iteratively refines the transformation by minimizing geometric distance between corresponding points. The result is an improved alignment along with the final estimated transformation matrix, which is visualized before and after refinement.

Before refinement
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/689127c2-94f3-4ea2-a246-4f330bef1958" />

After refinement
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/a050e500-a4d7-4cf3-a5e5-32e718a65950" />

---


Task 2
---

This script implements a fully customized 3D point-cloud registration pipeline using a least-squares formulation of the Iterative Closest Point (ICP) algorithm. The code constructs a manual optimization loop: each iteration computes nearest-neighbor correspondences, evaluates either point-to-point or point-to-plane residuals, and solves for the incremental 6-parameter rigid transformation using *scipy*’s **Levenberg–Marquardt** optimizer. The point clouds are generated from TUM RGB-D images, and normal vectors are used to enable point-to-plane minimization. Throughout the process, intermediate alignments and residual errors are visualized to monitor convergence. The final result is an estimated rigid transformation that successfully aligns the two RGB-D-derived point clouds, producing a refined geometric match after several optimization iterations. 

Initial visualization
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/21e7162b-fff1-4dea-a145-3e194d58e288" />


Intermediate visualization 1
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/5fcf2630-6627-47de-b181-d6df5d1d1dcf" />


Intermediate visualization 2
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/d82be703-a57a-45c5-b994-727933b93944" />


Final visualization
<img width="1854" height="1048" alt="image" src="https://github.com/user-attachments/assets/ee269deb-72c6-4ece-8b2a-4fd3e8c8824e" />

---


Task 3
---

This script computes the **minimum enclosing sphere** that contains all points from two RGB-D–derived point clouds. After generating the point clouds from TUM dataset images and downsampling them for efficiency, the program applies optimization strategie: a **constrained SLSQP formulation**, which minimizes the radius while enforcing that every point lies inside the sphere. Method output sphere parameters, along with detailed geometric statistics. The final result is a sphere mesh visualized alongside the two point clouds, showing the smallest sphere capable of enclosing their combined geometry. 

Initial sphere
<img width="1308" height="786" alt="image" src="https://github.com/user-attachments/assets/392f77b3-a383-492a-adc6-83c3d7d515f7" />


Intermediate sphere 1
<img width="1308" height="786" alt="image" src="https://github.com/user-attachments/assets/f5619182-ada1-452d-adce-8e2617368225" />


Intermediate sphere 2
<img width="1308" height="786" alt="image" src="https://github.com/user-attachments/assets/8ce1db0d-6c23-4162-b160-0619409de1d3" />


Intermediate sphere 3
<img width="1308" height="786" alt="image" src="https://github.com/user-attachments/assets/4736852c-2919-4e36-9b9e-47e6d2a14a43" />



Minimum enclosing sphere
<img width="1308" height="786" alt="image" src="https://github.com/user-attachments/assets/dff41a10-bd4b-4449-8865-1130451c2337" />

Volume: 73.407561 cubic units
Surface Area: 84.783496 square units
