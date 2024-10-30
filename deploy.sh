cd /home/$1
# Define the directory name where the repository should be cloned
dir="KoSimCSE-roberta"

# Check if the directory does not exist
if [ ! -d "$dir" ]; then
    # Directory does not exist, so clone the repository
    git clone https://huggingface.co/BM-K/KoSimCSE-roberta $dir
else
    echo "Directory '$dir' already exists."
fi

echo "Running 'make u' command"
make u