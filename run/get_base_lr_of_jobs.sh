for id in $(squeue -u $USER -o "%i" -h)
do

# Check if any file matches the pattern
if ls $GPS/output/*"${id}.out" 1> /dev/null 2>&1; then
    # Loop through matching files and grep for "base_lr:"
    for file in $GPS/output/*"${id}.out"; do
	echo ""
	echo "=== $id ==="
        grep "base_lr:" "$file"
    done
else
    :
    # echo "No output files for id ${id}"
fi

done

