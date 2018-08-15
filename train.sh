#run model script

output_dir=`grep outdir\ = $1| awk -F"=" '{print $2}'`
echo $output_dir
python run_model.py $1
