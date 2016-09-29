IN_FOLDER=/V/VPH-PRISM/breast_dataset/
OUT_FOLDER=/X/Userdata/Guido/slides/

SUBFOLDERS=("Benign/MRXS/" "DCIS/MRXS/" "IDC/MRXS/ Normal/MRXS/")


FILES=("T10-01877_slh" \
"T10-10006_mix" \
"T10-13039_fibrosis" \
"T10-13343_fibrosis" \
"T10-16995_fibrocystic" \
"T10-17745_pash" \
"T10-18454_mix" \
"T10-19590_pash" \
"T10-19606_fibrocystic" \ 
"T10-21271-I1-1" \ 
"T10-23694_ductHyperplasia" \
"T10-23695_ductHyperplasia" \
#"T10-25998-I3-1" \
#"T11-00097_fibrocystic" \
#"T11-01574_ductectasia"
#DCIS
"T10-00485-I-19-1-gr3" \
"T10-01977-I-5-1-gr3" \
"T10-02496-I-7-1-gr2" \
"T10-03264-I-8-1-gr2" \
"T10-10746-I-4-1-gr3" \
"T10-12789-I-5-1-gr3" \
"T10-17407-I-14-1-gr3" \
"T10-17600-I-12-I-gr3" \
"T10-19853-I-9-1-gr3" \
"T11-01751-I-5-1-gr2" \
"T11-02282-I-3-1-gr3" \
"T11-06455-I-5-1" \
#"T11-07260-I-10-1" \
#"T11-07727-I-4-1" \
#"T11-09026-I-7-1"
#IDC
"T10-10714-I-5-1" \
"T10-11091-I-14-1" \
"T10-13834-I-14-1" \
"T10-14101-I-15-1" \
"T10-18389-I-20-1" \
"T10-21498-I-9-1" \
"T10-22243-I-14-1" \
"T10-24083-I-12-1" \
"T11-02413-I-11-1" \
"T11-16052-I-19-1" \
"T11-18685-I-3-1" \
"T11-24422-I-13-1" \
#"T11-24497-I-4-1" \
#"T11-25105-I-12-1" \
#"T11-27354-I-7-1-gr2"
)

FILE=Benign/MRXS/T10-13343_fibrosis

for FILE in ${FILES[*]}
do
	if [ -f "${OUT_FOLDER}${FILE}.mrxs" ]
	then
		  echo "skipping ${FILE} (already done)"
	else
		FOUND=0
		for SUBFOLDER in ${SUBFOLDERS[*]}
		do
			if [ -f "${IN_FOLDER}${SUBFOLDER}${FILE}.mrxs" ]
			then
			  FOUND=1
			  echo copying $FILE from ${SUBFOLDER}
			  cp -r ${IN_FOLDER}${SUBFOLDER}${FILE}/ -d ${OUT_FOLDER}
			  cp -r ${IN_FOLDER}${SUBFOLDER}${FILE}.mrxs -d ${OUT_FOLDER}
			fi
		done
		
		if [ "$FOUND" -eq 0 ]
		then
			echo DID NOT FIND $FILE
		fi
		
	fi
done

