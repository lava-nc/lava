#!/usr/bin/env bash

echo "`git diff --check --cached | sed '/^[+-]/d'`" | while read -r line ; do
    if [ ! -z "$line" ]; then
        file="`echo $line | sed -r 's/:.*//'`"
        if [[ $file == *.rst ]] || [[  $file == *.ipynb ]]; then
            :
        else
            line_number="`echo $line | sed -r 's/.*:([0-9]+).*/\1/'`"
            backup_file="${file}.working_directory_backup"
            cat "$file" > "$backup_file"
            git checkout -- "$file" # discard unstaged changes in working directory
            sed -i "${line_number}s/[[:space:]]*$//" "$file"
            git add "$file" # to index, so our whitespace changes will be committed
            sed "${line_number}s/[[:space:]]*$//" "$backup_file" > "$file"
            rm "$backup_file"
            e_option="-e"
            echo $e_option "Removed trailing whitespace in \033[31m$file\033[0m:$line_number"
        fi
    fi
done

if [ ! -z `git diff --check --cached | sed '/^[+-]/d'`]; then
    exec git diff-index --check --cached $against --
fi