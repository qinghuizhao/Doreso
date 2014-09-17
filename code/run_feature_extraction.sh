
feat_list = $1

npydir = ./npyoutput
HPSSdir = ./HPSS


mkdir $npydir
mkdir $HPSSdir

for file in $feat_list; do
  python mfcc.py $file $npydir/mfcc/
  filename = 'basename $file'
  bin/HPSS $file $HPSSdir/$filename
done

for HPSS in $HPSSdir; do 
  python mfcc_gate.py $HPSS $npydir/mfcc_hpss/
done


