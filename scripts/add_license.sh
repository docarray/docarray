#!/bin/bash

LICENSE_TEXT=$(cat scripts/license.txt)  # Replace 'license.txt' with the actual path to your license file

# Iterate through all Python files
find docarray -name "*.py" -type f | while read -r file; do
    # Check if the license text is already in the file
    if ! grep -qF "$LICENSE_TEXT" "$file"; then
        # Prepend license notice to the file
        { echo "$LICENSE_TEXT"; cat "$file"; } > tmpfile && mv tmpfile "$file"
    else
        echo "License already present in $file"
    fi
done


# Iterate through all Python files
find tests -name "*.py" -type f | while read -r file; do
    # Check if the license text is already in the file
    if ! grep -qF "$LICENSE_TEXT" "$file"; then
        # Prepend license notice to the file
        { echo "$LICENSE_TEXT"; cat "$file"; } > tmpfile && mv tmpfile "$file"
    else
        echo "License already present in $file"
    fi
done