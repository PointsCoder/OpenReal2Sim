### This part is used to generate the affordance of the object.
### Include:
### 1. Grasp point generation.
### 2. Affordance map generation.
### TODO: Human annotation.
### The logic we use for grasp point generation here is: we select the frist frame of object-hand contact, compute the contact point, overlay the object and extract the 3D coordinate of the point.
### Be careful that we transfer this point to the first frame using pose matrices and extrinsics.
### TODO: This might be helpful to object selection in the first step.