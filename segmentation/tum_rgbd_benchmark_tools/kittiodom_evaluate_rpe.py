#-*- coding: utf-8 -*-
#!/usr/bin/python2
# Written by Geonuk LEE, 2023

from converter import *
from associate import *
from evaluate_rpe import *

if __name__ == "__main__":
    # ex) python2 tum_rgbd_benchmark_tools/kittiodom_evaluate_rpe.py kitti_odometry_dataset/poses/05.txt output.txt --verbose
    random.seed(0)

    parser = argparse.ArgumentParser(description='''
    This script computes the relative pose error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('groundtruth_file', help='ground-truth trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
    parser.add_argument('estimated_file', help='estimated trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
    parser.add_argument('--max_pairs', help='maximum number of pose comparisons (default: 10000, set to zero to disable downsampling)', default=10000)
    parser.add_argument('--fixed_delta', help='only consider pose pairs that have a distance of delta delta_unit (e.g., for evaluating the drift per second/meter/radian)', action='store_true')
    parser.add_argument('--delta', help='delta for evaluation (default: 1.0)',default=1.0)
    parser.add_argument('--delta_unit', help='unit of delta (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'s\')',default='s')
    parser.add_argument('--offset', help='time offset between ground-truth and estimated trajectory (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the estimated trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--save', help='text file to which the evaluation will be saved (format: stamp_est0 stamp_est1 stamp_gt0 stamp_gt1 trans_error rot_error)')
    parser.add_argument('--plot', help='plot the result to a file (requires --fixed_delta, output format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the mean translational error measured in meters will be printed)', action='store_true')
    parser.add_argument('--print_errors', help='print the error for each respective pose', action = 'store_true')
    args = parser.parse_args()

    #args.groundtruth_file= '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg3/rgbd_dataset_freiburg3_structure_texture_far/groundtruth2.txt'
    #args.estimated_file = '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Software/InfiniTAM-build/Files/Out/poses.txt'
    if args.plot and not args.fixed_delta:
        sys.exit("The '--plot' option can only be used in combination with '--fixed_delta'")
    
    traj_gt = read_trajectory(args.groundtruth_file)
    traj_est = read_trajectory(args.estimated_file)
    #sortedList = list(traj_gt.keys())
    
    #sortedList.sort();
    #traj_est = read_trajectory(args.estimated_file,True,sortedList)
    result = evaluate_trajectory(traj_gt,
                                 traj_est,
                                 int(args.max_pairs),
                                 args.fixed_delta,
                                 float(args.delta),
                                 args.delta_unit,
                                 float(args.offset),
                                 float(args.scale))
    
    stamps = numpy.array(result)[:,0]
    trans_error = numpy.array(result)[:,4]
    rot_error = numpy.array(result)[:,5]
    
    if args.save:
        f = open(args.save,"w")
        f.write("\n".join([" ".join(["%f"%v for v in line]) for line in result]))
        f.close()
    
    if args.verbose:
        print ("compared_pose_pairs %d pairs"%(len(trans_error)))

        print ("translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print ("translational_error.mean %f m"%numpy.mean(trans_error))
        print ("translational_error.median %f m"%numpy.median(trans_error))
        print ("translational_error.std %f m"%numpy.std(trans_error))
        print ("translational_error.min %f m"%numpy.min(trans_error))
        print ("translational_error.max %f m"%numpy.max(trans_error))

        print ("rotational_error.rmse %f deg"%(numpy.sqrt(numpy.dot(rot_error,rot_error) / len(rot_error)) * 180.0 / numpy.pi))
        print ("rotational_error.mean %f deg"%(numpy.mean(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.median %f deg"%numpy.median(rot_error))
        print ("rotational_error.std %f deg"%(numpy.std(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.min %f deg"%(numpy.min(rot_error) * 180.0 / numpy.pi))
        print ("rotational_error.max %f deg"%(numpy.max(rot_error) * 180.0 / numpy.pi))
    else:
        print (numpy.mean(trans_error))

    if args.plot:    
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        ax.plot(stamps - stamps[0],trans_error,'-',color="blue")
        #ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
        ax.set_xlabel('time [s]')
        ax.set_ylabel('translational error [m]')
        plt.savefig(args.plot,dpi=300)
        
    if args.print_errors:
        for i in range(0,stamps.shape[0]):
            print (stamps[i], trans_error[i], rot_error[i])
