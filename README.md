# AlignmentML
Using machine learning to assess read alignment quality.
Jacob S. Porter

## Directories

**code**
Python code used to extract features, train models, and analyze the results.

**models**
Trained scikit-learn models.

**labels**
Files that label the reads for each read file and each program.

**scripts**
Bash scripts used to run the programs.  These are included for reference.  They will not work generically because of hard coded directory locations.


## Code file descriptions

add_unmapped.py
Adds unmapped sequences to a label file.  File must be modified to run.

AlignmentTypeML_Plot.py
Generate plots for the confusion matrix.

AlignmentTypeML3.py
Train the models and give test set statistics for a read set.  This accepts a single file for features that represent the concatenated training and test set.

Constants.py
A set of contants for SeqIterator.py

convert_tsv.py
Converts the python style feature file into a tab separated value file.

countBFAST-Gap.py
Give the labels for a BFAST, BFAST-Gap, or BisPin SAM alignment file.

examine.py
Computes average alignment score and average edit distance.

extractFeatures.py
Gets features for machine learning from FASTQ reads.

getLabels_Bismark.py
Make the machine learning labels for Bismark.

predict.py
Use a trained model and a feature file to predict alignment categories.

ranks.py
Average the feature ranks and plot the box plots.

remove_wildcards.py
Iterates through a file and removes reads with wildcard characters.  This can be a useful utility.

SeqIterator.py
An iterator class for iterating through sequence record files.  Supports fasta, fastq, sam files.

## Workflow

1. Download reads.  Create indexes and perform alignments with alignment programs.
2. Extract features from reads with extractFeatures.py
3. Create labels for alignments with getLabels\_Bismark.py and countBFAST-Gap.py, etc.  May need add\_unmapped.py
4. Input labels and features into AlignmentTypeML3.py to perform machine learning training, feature ranking, and confusion matrix production.
5. Analyze results and produce plots with ranks.py, AlignmentTypeML_Plot.py, predict.py

## Example usage for AlignmentTypeML3.py

./AlignmentTypeML3.py -n 3 -s 2500000 -t 500000 -c Random,RF,MLP,LR -b -d /save_directory/ features.txt labels.txt 1> ml.out 2> ml.err

For help information on parameters use:
./AlignmentTypeML3.py --help 

## Notes
Reads were donwloaded from the NCBI trace archive.  I used the first 2.5 million reads for training, and the last 500k for testing.  The last 500k was used for filter analysis in select cases.

