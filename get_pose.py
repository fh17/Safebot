from estimater import *
from frameprocessing import reader
from YOLOv11seg import *
import argparse
from R4 import TransformationMatrix
from z_vector import *
import subprocess

start_time = time.time()  # start time measuring

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/data/mesh.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/data')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  K = np.loadtxt(f'{args.test_scene_dir}/cam_K.txt').reshape(3,3)
  W, H =  int(K[0, 2] * 2), int(K[1, 2] * 2)  # around (640, 480) #
  zfar = np.inf  # Update if necessary

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = reader(video_dir=args.test_scene_dir, H=H, W=W, shorter_side=None, zfar=np.inf)

  mask_generator = MaskGenerator()

  transformation = TransformationMatrix(f'{args.test_scene_dir}/R3.txt') #transformation matrix from base to home position

#.................................................................................................................................................................
#  depth, color = reader.get_frame(input) #here we have to put the pictures taken frm the robot home position (depth, color)
  color, depth = reader.get_frame()
#...................................................................................................................................................................

  if depth is not None and color is not None:

      # Process the saved image to get the mask
      cv2.imwrite(f'{args.test_scene_dir}/picformask.png', color)
      mask = mask_generator.process_image(f'{args.test_scene_dir}/picformask.png')

      cv2.imwrite(f'{args.test_scene_dir}/picmask.png', mask)

      # Continue with pose estimation and matrixtransformation

      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      # Retrieve and print the base-to-object transformation matrix
      print("Camera-to-Object Transformation Matrix:")
      print(pose)

      transformation.update_pose_matrix(pose)
      base_to_object_matrix = transformation.get_base_to_object()
      print("Base-to-Object Transformation Matrix:")
      print(base_to_object_matrix)
  else:
      logging.error("Failed to capture frames.")

  result = get_upward_and_position(pose)
  print(result)




  end_time = time.time() #end time measuring

  execution_time = np.array([end_time - start_time])
  np.savetxt(f'{args.test_scene_dir}/estimation_time.txt', execution_time, fmt='%f')



# So, as a final output we are getting the "upward" (z) axis of the box and its Geometry Center position vector.
# As input both, depth and RGB pictures (size 640x480) are provided;
# also the transformation matrx of the camera with respect to the base could be provided to get the results with respect to the base (more acurate)
# Detail: since we are working with a box, the axes x,y,z cannot be defined as such because of simetry.
# That's why in the last part of the code a function "get_upward_and_position" is called
# which defines z as the most "upward" vector (wrt the camera frame) always pointing up.


