import argparse
from pipeline import ivim_pipeline
from standardized.IAR_LU_biexp import IAR_LU_biexp

def main():
    parser = argparse.ArgumentParser(description="IVIM-MRI_Code-Collection")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the input medical image (NIfTI format)",
    )
    parser.add_argument(
        "--bvec",
        type=str,
        help="Path to the .bvec file",
    )
    parser.add_argument(
        "--bval",
        type=str,
        help="Path to the .bval file",
    )
    parser.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        default=[60, 60, 30],
        help="Voxel coordinates (x y z) to analyze (default: 60 60 30)",
    )
    parser.add_argument(
        "--direction",
        type=int,
        default=None,
        help="Gradient direction to analyze (default: None, uses all directions)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="IAR_LU_biexp",
        choices=["IAR_LU_biexp"],
        help="IVIM fitting algorithm (default: IAR_LU_biexp)",
    )

    args = parser.parse_args()

    # Convert voxel list to tuple
    voxel = tuple(args.voxel)

    # Run the analysis pipeline
    result = ivim_pipeline(
        bvec_path=args.bvec,
        bval_path=args.bval,
        data_path=args.data_path,
        voxel=voxel,
        direction=args.direction,
        algorithm=args.algorithm,
    )

    print("IVIM Fit Results:")
    for key, value in result.items():
        print(f"{key}: {value}")



if __name__ == '__main__':
    main()