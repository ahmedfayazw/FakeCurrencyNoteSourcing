# Fake Currency Note Sourcing

Identify Fake Indian currency notes and store them in Database. when a new note is found compare it with database and return top matching results.

## Piracy and Open-Sourcing

**As this is a Goverment project certain details have been disclosed. Only the algorithms used have been listed and code is not.**

## Description of the Problem

- The main aim of this project is to identify similar fake notes, and trace their origin. 
- This eases the work of the employees, as they have to manually process a large number of notes and find similar notes

## Literature Survey

- Previously the system uses man power to trace the origin of similar notes.
-  After this the project has been taken and a software was created to store and retrieve features of fake notes. 
- Currently the software is being upgraded. 
- The work of employees is to store the features of already existing fake notes and when they input a new case of fake note, the software returns the top matching results of the fake note.

## Objectives and Scope of the project

### The main objective of the project:

- Tracing Source of fake currency notes
- Comparing Different features of fake notes for accurate results
-  To embed the data into a database.
- For user to access the database and save a copy of it for future use.
- For user to compare many samples at a time.

### Scope of the Project

- The fake notes are fed to the software.
- The software compares the features of fake note with all the fake notes stored in Database.
- The Software returns the top matching results of the fake note.
- User cross references the top results and confirms the matching results.
- If the source already exists, then the fake note is traced, if not the features of this new fake note is stored in Database for future use.

## Solution Methodology

- This Software is developed using Python's image processing libraries like openCV and numPy packages, and using PyQt as a front-end and SQlite for database. 
- This Software uses a algorithm called matchTemplate which detects the features of an Currency Note that are useful in tracing the source. 
- Then the features are compared with the fake notes already available in the database to see if a match is available. After processing, the user can also store the data in an Excel file for easy accessing

### Analysis And Design

### Platform and Tools 

#### Hardware 

- 4GB Ram
- 64-bit Operating System
- 300 dpi Scanner

#### Software 

- Python 3.9
- Pycharm Community Edition
- PyQt5 GUI Designer
- SQLite for Database
- Excel For Database viewer

#### Libraries Used

- Numpy
- OpenCv
- ImageSlicer
- Pytesseract
- SQlite
- Os
- PyQt5
- Sys
- Time
- Pandas

### Activities

1. Number Extraction
2. Updating the features of counterfeit notes to database.
3. Counterfeit note extraction by searching database
4. Currency note comparison between any two notes

### Results

#### Expected Results

1. Serial Number detection
2. Features of the counterfeit notes should be updated to the database
3. Top matches of the new case counterfeit note to be returned
4. Similarities between any two notes to be returned

#### Observed Results

1. Serial  Number  detection  is  done  at  90  percent  efficiency,  some  results   return are mismatched like, the alphabet ’k’ is detected as numeric number.  This is overcame by entering the serial number manually by the user
2. Features of the counterfeit notes are successfully updated to the database
3. Top matches of the new case counterfeit notes are successfully returned
4. The similarities of any two notes are returned with 90 percent efficiency, some features do not resemble much even if 2 original notes are given

### Algorithms Used

#### Image Pre-processing 

1. Grey-scale :  Converting into Black and white image
2. Gaussian-Blur :  Removes the high-frequency components
3. CannyEdgeDetection : Uses a multi-stage algorithm to detect a wide rangeof edges in images,•Dilation :  increases the object area
4. Erosion :  Decreases the Object area,•Contours :  Tool for shape analysis and object detection and recognitionusing a curve joining all the continuous points
5. PerspectiveTransform :  change the perspective of the image

### Serial Number Detection 

1. Pytesseract:  Optical Character Recognition (OCR) tool
2. MatchTemplate, CV2 :  detect the area where the serial number is located•Cropping ; crop the serial number area
3. Threshold :  isolating the serial numbers

### Updating database 

1. SQLITE3 :  Database used to store, retrieve data
2. MatchTemplate, CV2 :  detect the area where the features of the note arelocated
3. MatchTemplate, CV2 :  find the similarity percentage between features

### Counterfeit comparison 

1. MatchTemplate, CV2 :  detect the area where the features of the note arelocated
2. MatchTemplate, CV2 :  find the similarity percentage between features

### App Specification

#### Serial Number Extraction

1. Select the appropriate denomination and click Choose Folder button to choose the currency notes as many as needed from folder
2. To view the selected images, click Review button
3. Finally click Process button to retrieve the serial numbers
4. On  clicking  the  Process  button,  Phony  Finder  displays  the  serial numbers extracted from the currency notes in the Table below
5. The Reset button clears the input given by the user

#### Update Database 

1. Select the appropriate denomination and click Choose Folder button to choose currency notes as many as needed from folder 
2. To view the selected images, click Review buttons
3. Click Process button to extract features from the selected notes and push the data into database 
4. On clicking the Process button, Phony Finder displays the quantity of the database in the text box present below 
5. The Reset button clears the input given by the user

#### Counterfeit Extraction 

1. Select the appropriate denomination and click Choose Folder button to choose a currency note from folder
2. To view the selected images, click Review button
3. Click Process button to extract and compare features of the selected note
4. Then select the no of matches as needed and click Top Results button to view the top matches of the selected note
5. On clicking the Top Results button,  Phony finder displays the topmatches of the selected note from the database in the Table
6. The Reset button clears the input given by the user

#### Counterfeit Compare 

1. Select the appropriate denomination and click Choose Folder button to choose any two currency notes from folder
2. To view the selected images, click Review buttons
3. Click  Process  button  to  extract  and  compare  features  of  the  twoselected notes
4. On clicking the Process button, Phony Finder displays the similaritypercentage of the selected currency notes in the text boxes presentbelow
5. The Reset button clears the input given by the user

### Back-end 

#### Serial Number Extraction 

1. The serial number area of the selected note is cropped
2. Then the image is processed with ’Pytesseract’ algorithm, an opticalcharacter recognition tool.
3. the  serial number  is detected,  stored in  table and  displayed  to theuser

#### Update Database 

1. The features of the notes are cropped by ’MatchTemplate, CV2’ al-gorithm.
2. These features are compared with a single template note to returnthe properties of the features of the notes.
3. The  properties  of  the  notes  are  then  stores  into  the  database  forfuture use•Counterfeit Extraction :
4. The features of the notes are cropped by ’MatchTemplate, CV2’ al-gorithm.
5. These  features  are  compared  with  the  already  existing  data  in  thedatabase.
6. The top results of the matches are displayed•Counterfeit Compare :
7. the features of the notes are cropped by ’MatchTemplate, CV2’ al-gorithm.
8. These features are compared directly.
9. The results are displayed

