import os
import SimpleITK as sitk
import numpy as np
#%%
# Create a 2D rigid transformation, write it to disk and read it back.
basic_transform = sitk.Euler2DTransform()
basic_transform.SetTranslation((1,2))
basic_transform.SetAngle(np.pi/2)

full_file_name = ('euler2D_example.txt')

sitk.WriteTransform(basic_transform, full_file_name)

# The ReadTransform function returns an sitk.Transform no matter the type of the transform
# found in the file (global, bounded, composite).
transform = sitk.ReadTransform('euler3D.txt')
print(transform)
# print('Different types: '+ str(type(read_result) != type(basic_transform)))

#TODO: what's the center of rotation