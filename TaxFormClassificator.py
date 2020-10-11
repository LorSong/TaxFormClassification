import os
import cv2
import numpy as np

from scipy import ndimage
from PIL import Image
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz

from tesserocr import PyTessBaseAPI, PSM

import tensorflow as tf
if tf.__version__ != "2.3.0":
    print("TF version is not 2.3.0, behavior may not be correct")
   

class TaxFormClf():
    """
    Class for predicting tax form documents. Classes are:
    ['2НДФЛ', '3НДФЛ', 'Форма банка', 'Неизвестный документ'].
    ---------------------
    Main method - 'load_predict'. Takes path to a folder with images you want to
    predict. For additional details see docstring for this method.
    ---------------------
    parameters: model_path - path to trained tensorflow model
    ---------------------
    Note! OCR operations are slow, can take up to 20 seconds for an image.
    """
    
    def __init__(self, model_path='model'):
        self.class_names = ['2НДФЛ', '3НДФЛ', 'Форма банка', 
                                        'Неизвестный документ']
        # Get text tokens for string matching
        self.tokens = self._get_tokens()
        # Loading TF model
        self.model_path = model_path
        try:
            print('[INFO] Loading TF model. tf must be imported')
            self.model = tf.keras.models.load_model(self.model_path)
        except:
            raise 'Failed to load model, check model_path'
                
                
    def predict(self, path):
        """
        Takes path to a folder with documents. Processes them and
        returns predictions in form of a dictionary :
        {filename_1.jpg: 'НДФЛ2', filename_2.jpg: 'НДФЛ3'}
        -----------------
        Note! OCR operations are slow, can take up to 20 seconds for image.
        -----------------
        Patameters:
        -----------------
        path: string path for os.listdir function
        
        After execution sets Attributes:
        ---------------------------------------
        pred_labels: array with class labels, that correspond to 
        'class_names' attribute.
        
        pred_probas: 2d array with predicted probabilities, where columns 
        correspond to class_names attribute
        
        proc_images: list with images (np.arrays)
        texts: list with extracted texts (stings)
        img_names: list with image names (stings)
        rotations: list with performed rotations (90, 180, 270)
        cnn_probas: 2d array with predicted probabilities of CNN model
        ocr_probas: 2d array with predicted probabilities by OCR and string matching
        """
        # Also saving rotations attribute
        proc_imgs, texts, img_names = self._load_process_images(path)
        
        # Performing predictions and saving attributes
        preds = self._form_predictions(proc_imgs, texts, img_names)
    
        print('[INFO] Finished')
        return preds
      
    def _load_process_images(self, path):
        """
        Takes path to a folder with images, loads them, preprocesses
        and extracts text.
        ----------
        Returns lists with: processed_images, texts, image_names

        """
        # Loading
        images, img_names = self._read_images_from_folder(path)
        # Preprocessing
        images = self._preprocess_images(images, img_names)
        # Fix orientation and extract text
        images, texts = self._extract_text_fix_orient(images, img_names)  
        return images, texts, img_names
    
    def _form_predictions(self, proc_imgs, texts, img_names):
        """
        Takes outputs of _load_process_images method and forms predictions.
        
        Returns predictions in form of a dictionary :
        {filename_1.jpg: 'НДФЛ2', filename_2.jpg: 'НДФЛ3'}
        """
        # Also saves ocr_preds, cnn_preds attributes
        self.pred_probas = self._predict_proba(proc_imgs, texts)
        self.pred_labels = np.argmax(self.pred_probas, axis=1)
        # Creating dictionary with predictions
        pred_classes = [self.class_names[i] for i in self.pred_labels]
        preds = dict(zip(img_names, pred_classes))
        # Saving attributes
        self.proc_images = proc_imgs
        self.texts = texts
        self.img_names = img_names
        return preds
    
    def _predict_proba(self, images, texts):
        """
        Takes processed images and returns a list with probabilities, 
        which correspond to class_names attribute.
        
        method (string): 'BOTH' (default), 'CNN', 'OCR'. 
        Specifies what methods to use when making predictions
        
        model_path: path, where CNN model is located
        """
        print('[INFO] Making predictions using extracted text')
        ocr_probas = self._OCR_predict(texts)
        print('[INFO] Making predictions using CNN model')
        cnn_probas = self._CNN_predict(images)
        ens_probas = self._combine_preds(cnn_probas, ocr_probas)
        
        self.ocr_probas = ocr_probas
        self.cnn_probas = cnn_probas
        return ens_probas
        
    def _OCR_predict(self, texts):
        """
        Takes texts and returns probabilities of them belonging to a class.
        Returns np.array, where columns correspond to class_names attribute.
        """
        scores = []
        for text in texts:
            scores.append(self._score_text(text))
        # Transform scores into probabilites
        scores = np.array(scores) / 100
        scores = scores / scores.sum(axis=1, keepdims=1)
        return scores

    def _CNN_predict(self, images):
        """
        Takes tensorflow model and images.
        Returns predicted probabilities.
        """
        # Resize images
        res_imgs = []
        for image in images:
            image = 255 - image
            image = np.expand_dims(image, -1)
            resized = tf.image.resize_with_pad(image, 224, 224)
            res_imgs.append(resized)
        # Form batched dataset  
        ds = tf.data.Dataset.from_tensor_slices((res_imgs)).batch(16)  
        
        preds = self.model.predict(ds)
        return preds
      
    def _score_text(self, text):
        """
        Looks for matches in a text with tokens for all classes. 
        Returns a list with scores (unnormalized).
        """
        tokens = self.tokens
        # Get scores for all classes
        # NDFL 2 has 2 sets of tokens (different forms)
        ndfl2_score = max(self.__score_class(tokens['ndfl2_1'], text),
                      self.__score_class(tokens['ndfl2_2'], text))
        ndfl3_score = self.__score_class(tokens['ndfl3'], text)
        bank_score = self.__score_class(tokens['fbank'], text)
        
        score = [ndfl2_score, ndfl3_score, bank_score]
        # Check if it's confident in any class
        best_doc = max(score)
        # Other score could be parameterized
        other_score = min(115 - best_doc, 100)
        score.append(other_score)
        return score   
    
    
    def _combine_preds(self, preds_cnn, preds_ocr, fix_other=True, 
                       cnn_w=0.2, oth_ocr_cnn_conf=0.8, other_cnn_w=0.15):
        """
        Combines predictions.
        
        fix_other: Bool. If true, examples that estimator predicted as 
        'other' class, will be estimated mainly by other estimator.
        
        cnn_w: Weight of CNN impact on predictions 
        oth_ocr_cnn_conf: Level of CNN confidence on examples where OCR
        classified 'other' class. If confidence is higher than this value, 
        used CNN prediction.
        other_cnn_w: Weight of CNN impact on predictions, where CNN
        classified 'other' class.
        """
        if fix_other:    
            preds = preds_ocr.copy()
            # Take labels where cnn or ocr predicted other
            oth_ocr_mask = np.argmax(preds_ocr, axis=1) == 3
            oth_cnn_mask = np.argmax(preds_cnn, axis=1) == 3

            oth_ocr_idx = np.where(oth_ocr_mask)
            oth_cnn_idx = np.where(oth_cnn_mask)
            
            # Where they both predicted document
            not_doc_mask = np.logical_or(oth_ocr_mask, oth_cnn_mask)
            doc_idx = np.where(np.logical_not(not_doc_mask))
            
            # Where OCR predicted 'other class' use CNN if its confident
            for idx in oth_ocr_idx[0]:
                if max(preds_cnn[idx]) > oth_ocr_cnn_conf:
                    preds[idx] = preds_cnn[idx]
            
            # Weighting where CNN predicted 'other class' 
            preds[oth_cnn_idx] = (preds_cnn[oth_cnn_idx] * other_cnn_w + 
                    preds_ocr[oth_cnn_idx]) / (other_cnn_w + 1.) 
            
            # Weighted sum for predicted documents
            preds[doc_idx] = (preds_cnn[doc_idx] * cnn_w + 
                                      preds_ocr[doc_idx]) / (cnn_w + 1.)
        else:
            preds = (preds_cnn * cnn_w + preds_ocr) / (cnn_w + 1.)
        return preds
           
    
    @staticmethod
    def __score_class(tokens, text):
        """
        Scores text for a class with fuzzyfuzzy library that uses Levenshtein 
        Distance to calculate difference between tokens and text. Each token 
        estimated separately.
        """
        score = 0
        for c, token in enumerate(tokens):
            score += fuzz.partial_ratio(token, text)
            # Adding weight to the first token
            if c == 0 :
               score *= 3
        return score / (c + 3) 
    
    @staticmethod
    def _read_images_from_folder(folder):
        """
        Takes path to folder and loads all images as cv2 image.
        Returns lists of images and their filenames.
        """
        images = []
        filenames = []
        all_files = [file for file in os.listdir(folder) 
                             if os.path.isfile(os.path.join(folder, file))]
        for filename in all_files:
            if filename.endswith('.pdf'):
                pages = convert_from_path(os.path.join(folder, filename), 300)
                img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            else:
                byte_im = np.fromfile(os.path.join(folder, filename), dtype=np.uint8)
                img = cv2.imdecode(byte_im, cv2.IMREAD_COLOR)
            if img is not None:
                if len(img.shape) != 3:
                    print(folder, filename, 'was read incorrectly, skipping')
                images.append(img)
                filenames.append(filename)
        if len(os.listdir(folder)) != len(images):
            print('[INFO] Not all files were considered to be images')
        
        if not images:
            raise 'Error! No images were found, check path'
        
        return images, filenames
        
    def _preprocess_images(self, images, img_names):
        """
        Takes a list with loaded images in np.array format and their filenames,
        Returns a list of processed images in the same format
        
        During preprocessing trying to align image, fix skew and enhanse
        readability for OCR.
        ----------
        images : np.array of images, preferably read with cv2.imread()
        img_names : list with their filenames
        """
        
        processed_im = []
        print('[INFO] Found {} images. Prerocessing...'.format(len(images)))
        for image, img_name in zip(images, img_names):  
            # Convert to gray
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Aligning to screen and fixind shadow
            img = self._align_to_screen(img, img_name)
            # Fixing skew
            img = self._fix_skew(img)
            processed_im.append(img)
        print("[INFO] Finished preprocessing, starting OCR")
        return processed_im
            
    def _extract_text_fix_orient(self, images, im_names):
        """
        Extracts text from the image using pytesseract
        Also fixes incorrect image orientation.
        Returns (image, text).
        """
        target_size = 4000
        rot_images, texts = [], []
        self.rotations = []
        
        print("Started OCR")
        for i, (img, im_name) in enumerate(zip(images, im_names)):
            self.__print_ocr_status(i, im_name, len(im_names))
            
               
            # Increase image size for OCR
            image = img.copy()
            if image.shape[0] < (target_size - 500):
                fx = fy = target_size / image.shape[0]
                image = cv2.resize(image, None, 
                                         fx=fx, fy=fy, 
                                         interpolation=cv2.INTER_CUBIC)
            # tesserocr expects PIL image        
            image = Image.fromarray(image)
            
            # Determine orientation and extract text
            with PyTessBaseAPI(lang='rus', psm=PSM.AUTO_OSD) as api:
                api.SetImage(image)
                api.Recognize()
                it = api.AnalyseLayout()
                orientation, direction, order, deskew_angle = it.Orientation()
                
                angle = 0
                # Fixing orientation
                if orientation != 0.:
                    angle = 360 - orientation * 90
                    # also rotating not enlarged image
                    img = ndimage.rotate(img, angle)
                    image = image.rotate(angle, expand=True)
            # Extracting text        
            with PyTessBaseAPI(lang='rus', psm=6) as api:
                api.SetImage(image)
                text = api.GetUTF8Text()
                
            # Check if found text belongs to any class
            score_1 = self._score_text(text)[3]
            # If score of 'other document' is big, try to rotate and redo OCR
            if score_1 > 55:
                angle = angle + 180
                image1 = image.rotate(180, expand=True)
                with PyTessBaseAPI(lang='rus', psm=6) as api:
                    api.SetImage(image1)
                    text1 = api.GetUTF8Text()
                score_2 = self._score_text(text1)[3]
                # If score is improved rotate original image and keep the text
                if score_2 < 45:
                    img = ndimage.rotate(img, 180)
                    text = text1
                    
            self.rotations.append(angle)
            
            rot_images.append(img)
            texts.append(text) 
        
        return rot_images, texts  
    
    @staticmethod 
    def _fix_shadow(gray):
        """
        Fixes uneven brightness
        """
        coeff = np.log(max(0, (245 - gray.mean())) / 125 + 1) + 1
        gray = cv2.multiply(gray, coeff)

        dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        gray = 255 - cv2.absdiff(gray, bg_img)
        return gray
    
    @staticmethod    
    def _fix_skew(image):
        """
        Fixes small rotations of image
        """
        # Flipping colors, thresholding
        inv_image = cv2.bitwise_not(image)
        thresh = cv2.threshold(inv_image, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Taking all white pixels coordinates and compute rotated bounging box
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Fixing angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotating the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)   
        
        # print("[INFO] Fixed skew of {:.3f} degrees".format(angle))
        return rotated
    
    
    def _align_to_screen(self, img, img_name=None, fix_shad=True, repeat=False):
        """
        Aligns documents to screen (top down view without background).
        This fixes problem of photos, which may have background and wrong perspective. 
        Can be done second time automatically, with different preprocessing steps, 
        if the first result failed.
        If alignment failed, will return original image.
        """
        gray = img.copy()
        if fix_shad:
            gray = self._fix_shadow(gray)
            
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find contours and sort for largest contour
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None

        for c in cnts:
            # Perform contour approximation
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                displayCnt = approx
                break
        if displayCnt is not None:
            # Obtain a consistent order of the points
            rect = self.__order_points(displayCnt.reshape(4, 2))
            # Obtain birds' eye view of image
            warped = self.__four_point_transform(gray, rect)
        else:
            if repeat:
                # print('[INFO] Failed to align {}, skipping'.format(img_name))
                return self._fix_shadow(img)
            # Trying again, whithout fixing shadow
            warped = self._align_to_screen(img, img_name, fix_shad=False, repeat=True)
        area_change = (img.shape[0] * img.shape[1]) / (warped.shape[0] * warped.shape[1])
        # Checking if warping were successful
        if ((area_change > 3) or (np.all(warped == warped[0]))):
            # print('[INFO] Failed to align {}, skipping'.format(img_name))
            if repeat:
                img = self._fix_shadow(img)
            return img
        
        return warped
    
    @staticmethod
    def __four_point_transform(image, rect):
        """ Used in _align_to_screen method """
        # Unpack points
        (tl, tr, br, bl) = rect
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Specifying points for top-down view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def __order_points(pts):
        """ Used in _align_to_screen method """
        # Initialzie a list of coordinates
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # Compute the difference between the points
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates
        return rect
    
    @staticmethod
    def __print_ocr_status(iteration, file, total):
        """ Printing status bar """
        end = "" if iteration < total else "\n"
        print("\r {}".format(' '.join([' '] * 40)), end="") # Clearing line
        print("\rProcessing {}, {}/{}".format(file, iteration + 1, total), 
                                                          end=end)
    
    @staticmethod
    def _get_tokens():
        """This text used to score a document"""
        
        tokens = {
    'ndfl2_1': ['СПРАВКА О ДОХОДАХ И СУММАХ НАЛОГА ФИЗИЧЕСКОГО ЛИЦА',
        'Приложение № 5 к приказу ФНС России от 02.10.2018 № ММВ-7-11/566@',
        '1. Данные о налоговом агенте',
        '2. Данные о физическом лице - получателе дохода',
        '4. Стандартные, социальные и имущественные налоговые вычеты',
        '5. Общие суммы дохода и налога'],
    'ndfl2_2': ['СПРАВКА О ДОХОДАХ ФИЗИЧЕСКОГО ЛИЦА',
        'форма 2-НДФЛ',
        'Форма по КНД 1151078',
        'Признак номер корректировки в ИФНС (код)',
        '1. Данные о налоговом агенте',
        'Форма реорганизации (ликвидации) (код)',
        '2. Данные о физическом лице - получателе дохода',
        'Код документа, удостоверяющего личность:',
        '3. доходы облагаемые по ставке',
        '5. Общие суммы дохода и налога',
        'Наименование и реквизиты документа, подтверждающего полномочия',
        'Уведомление, подтверждающее право на социальный налоговый вычет'],
    'ndfl3': ['форма 3-НДФЛ',
        'Форма по КНД 1151020',
        'Налоговая декларация', 
        'по налогу на доходы физических лиц',
        'Сведения о налогоплательщике',
        'Сведения о документе, удостоверяющем личность:',
        'Статус налогоплательщика',
        'Декларация составлена на',
        'Достоверность и полноту сведений, указанных в \
                    настоящей декларации, подтверждаю',
        'Наименование документа, подтверждающего полномочия представителя',
        'Заполняется работником налогового органа', 
        'Сведения о представлении декларации'],
    'fbank': ['для получения кредита',
        'СПРАВКА О ДОХОДАХ', 
         'В том, что он (она) постоянно работает',
         'в должности'
         'Руководитель',
         'Главный бухгалтер',
        'Справка действительна для предоставления в \
        Банк в течение 30 календарных дней с даты ее выдачи',
        'за последние месяцев составил',
         'Если стаж на последнем месте работы более 3-х \
        но менее 12-ти месяцев - документы предоставляются за \
        фактически отработанный на последнем месте \
        работы период, но не менее чем за 3 полных месяца.',
        'Увольнения по инициативе работодателя не планируется',
        'удержаний',
        'Период Начислено',
        'Удержано к выплате']}
        
        return tokens
    
def main():  
    pass

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    