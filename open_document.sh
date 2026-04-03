BASE_DIR='/mnt/Database Storage/http/capstone'

xdg-open "$BASE_DIR/film_copyright/$1/$1.pdf" &


# ls "$BASE_DIR/qwen_ocr" | grep $1.\*\.txt | sort -V | while read fname
# do
#     CONTENT="$CONTENT"\n$(cat "$BASE_DIR/qwen_ocr/$fname")
# done

# echo $CONTENT