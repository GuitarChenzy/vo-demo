DATA_ROOT=/media/czy/DATA/Share
TRAIN_SET=$DATA_ROOT/Kitti/kitti_256/
python train.py $TRAIN_SET \
--resnet-layers 18 \
-b4 -s0.1 -c0.5 --epoch-size 1000  \
--with-ssim 1 \
--with-pretrain 1 \
--log-output --with-gt \
--name resnet18_depth_256