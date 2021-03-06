FILE=$1

echo "Note: available models are kitti_depth, nyu_depth, and nyu_surfacenormal"
echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/models/TransDepth/${FILE}_pretrained.tar.gz
TAR_FILE=./checkpoints/${FILE}_pretrained.tar.gz
TARGET_DIR=./checkpoints/${FILE}_pretrained/

wget -N $URL -O $TAR_FILE

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./checkpoints/
rm $TAR_FILE