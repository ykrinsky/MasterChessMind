import cv2
import imutils 
import numpy as np
import matplotlib.pyplot as plt
import heapq
from transformers import ViTImageProcessor
import torch
from transformers import ViTForImageClassification


class Board(object):
    '''
    possible values for the board:
    None: empty square
    'PW': white pawn
    'PB': black pawn
    'NW': white knight
    'NB': black knight
    'BW': white bishop
    'BB': black bishop
    'RW': white rook
    'RB': black rook
    'QW': white queen
    'QB': black queen
    'KW': white king
    'KB': black king
    '''
    def __init__(self):
        self.board_images = [[None]*8 for _ in range(8)]
        self.board_labels = [['']*8 for _ in range(8)]

def smart_thresholding(image):
    '''
    idea of the algorithm is to determine if there is an object and a background in the image. 
    we will create a grayscale histogram, and then find the best "valley" in the histogram and choose this as our thresholding. 
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # find the best valley in the histogram
    w = np.zeros(len(hist)-2)
    for i in range(1, len(hist)-1):
        v1  = sum(hist[:i])
        v2 = sum(hist[i:])
        w[i-1] = np.abs(v1-v2) 
    threshold = np.argmin(w)
    return threshold, w[threshold]
    # sc_h = np.histogramdd(np.array(imf_s), bins='auto')

# def is_empty(image, other_images):
#     # each chess board must have many

class PiecesModel(object):
    def __init__(self):
        model_name_or_path = 'chessboard_identifier/vit-base-chess'
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTForImageClassification.from_pretrained(model_name_or_path).to(self.device)
       
    def identify_piece(self, images):
        # test_image_6_7.png should be a pawn
        # first filter out the empty squares
        pieces_images = []
        pieces_indices = []
        for i, img in enumerate(images):
            if i == 8*7 + 5:
                print("testing image 6,7")   
            t, diff =  smart_thresholding(img)
            # See if there are two "distinct" blobls of color in the image.
            # TODO: Note that this does not check for distinct blobs, if i want to do this i also need to check for high standard variation in the image.

            if diff > img.shape[0]*img.shape[1] *0.6:
                print("empty", t, diff)
                continue
            pieces_images.append(img)
            pieces_indices.append(i)
            # TODO - there is even further improvemnt that i can do here - to check that there is an object at the center of the image.
            # I can do this by searching for "solid" objects and see if there is a large one at the center of the image. 

        pieces_images = np.array(pieces_images)
        inputs = self.processor(pieces_images, return_tensors='pt').to(self.device) 

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_labels = []
        for i in range(len(logits)):
            predicted_class_idx = logits[i].argmax(-1).item()
            predicted_labels.append(self.model.config.id2label[predicted_class_idx])
        
        final_predict_labels = []
        for i in range(len(images)):
            if i in pieces_indices:
                final_predict_labels.append(predicted_labels.pop(0))
            else:
                final_predict_labels.append("empty")

        return final_predict_labels




class BoardIdentifier(object):
    def __init__(self, image):
        self.image = image
        self.pieces_model = PiecesModel()
        self.debug = True
        self.board = Board()
    def get_board_positions(self):
        board = Board()
        # splitted_board = self.get_splitted_board(image)
        self.identify_board()
        # should initiate self.img_canny and self.splitted_board
        # TODO: decide if a pice is black or white.
        # TODO: decide if there is even a piece in the square or not.

        np_board = np.array(self.splitted_board)
        flattened_board = np_board.reshape([np_board.shape[0]*np_board.shape[1]] + list(np_board.shape[2:]))
        labels = self.pieces_model.identify_piece(flattened_board)
        if self.debug:
            display_board_image(self.splitted_board, labels)
        for i in range(8):
            for j in range(8):
                board.board_images[i][j] = self.splitted_board[i][j]
                board.board_labels[i][j] = labels[i*8 + j]
        return board

    # def get_splitted_board(self, image):
    #     (top_left, bottom_right, delim) = self.identify_board(image, image)
    #     splitted_board = self.split_chess_board(image, top_left, bottom_right, delim)
    #     return splitted_board

    def identify_board(self):
        # For this we will convert the image into black and white - to make it easier to identify borders.
        
        # resized = imutils.resize(image, width=300)
        # ratio = image.shape[0] / float(resized.shape[0])
        # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ratio = 1
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        """
        Run canny edge detector - 
        https://www.youtube.com/watch?v=sRFM5IEqR2w&ab_channel=Computerphile

        Essentially takes the output of a sobel operator and does some cleaning up.
        a. thins the edges
        We do this by finding local maximum of the edge in a mask, eliminating what is not the local maximum. this way we only keep the
        edge pixels on the "direction" of the edge
        b. does a 2-level thresholding (hysteresis) -
            i. Set two thresholds - high and low.
            ii. If the pixel is above the high threshold, it will be automaticvally included
            iii. If the pixel is below the low threshold, it will be automatically excluded
            iv. If the pixel is between the two thresholds, it will be included only if it is connected to a pixel above the high threshold
        """
        t_lower = 60  # Lower Threshold 
        t_upper = 200  # Upper threshold 
        aperture_size = 3  # Aperture size 
        
        img_canny = cv2.Canny(gray, t_lower, t_upper,  
                    apertureSize=aperture_size, L2gradient =True) 
        
        (top_left, bottom_right, delim) = detect_chess_board(img_canny, self.image)
        # s_image = draw_chess_frame(image, (top_left, bottom_right, delim))
        # cv2.imshow("s_image", s_image)
        # cv2.waitKey(0)
        self.splitted_board = split_chess_board(self.image, top_left, bottom_right, delim)
        self.img_canny = img_canny
        display_board_image(self.splitted_board)
        # save_board_images(splitted_board, "test_image")
        print("drawing square - ", (top_left, bottom_right, delim))
        
        return (top_left, bottom_right, delim)

def draw_on_image(mask_img, original_image, color):
    # for every pixel which is not zero in the mask_image, draw a point in the original image - 
    original_image[np.nonzero(mask_img)] = color
    return original_image

def display_subimages(s, img1, img2):
    i,j,l = s
    sub1 = img1[i:i+l+1, j:j+l+1]
    sub2 = img2[i:i+l+1, j:j+l+1]
    plt.subplot(121)
    plt.imshow(sub1)
    plt.subplot(122)
    plt.imshow(sub2, cmap='gray')
    plt.show()


def draw_square(img, s):
    ul,lr = s
    out_img = cv2.rectangle(img, (ul[1], ul[0]), (lr[1], lr[0]), (0, 255, 0), 2)
    return out_img

def draw_chess_frame(img, b):
    top_left, bottom_right, delim = b
    for i in range(8):
        for j in range(8):
            if (i+j) % 2 == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            img = cv2.rectangle(img, (top_left[1] + i*delim, top_left[0] + j*delim), (top_left[1] + (i+1)*delim, top_left[0] + (j+1)*delim), color, 2)
    return img
# def draw_square(img, s):
#     (i,j),l = s
#     out_img = cv2.rectangle(img, (j, i), (j+l, i+l), (0, 255, 0), 2)
#     return out_img


def find_even_spread_dots(arr, dist_mistake, num_dots=7):
    # assume array is sorted 
    for i in range(0, len(arr)-(num_dots-1)):
            distances = [arr[j+1] - arr[j] for j in range(i, i+num_dots-1)]
            if max(distances) - min(distances) > dist_mistake: 
                continue
            suspected_square_delim = sum(distances)//len(distances)
            yield i, suspected_square_delim

def verify_even_spread_dots(arr, dist_mistake, suspected_square_delim):
    distances = [arr[j+1] - arr[j] for j in range(0, len(arr)-1)]
    if max(distances) - min(distances) > dist_mistake: 
        return False
    if sum(distances)//len(distances) < suspected_square_delim+ dist_mistake and sum(distances)//len(distances) > suspected_square_delim - dist_mistake:
        return True
    return False


class Cross(object):
    def __init__(self, i, j):
        self.row = i
        self.col = j
    def __str__(self) -> str:
        return f"({self.row}, {self.col})"
    def __repr__(self) -> str:
        return f"({self.row}, {self.col})"

def detect_chess_board(img, original_image):
    '''
    receive a canny image. my target is to find the crosses which are created when 4 squares of the chess board are connected.
    sometimes the crosses are not well connected in the canny image, so the approach is to use Hit-And-Miss morphological operation
    to find the crosses, but to also allow for errors in the middle of the crosses.

    After doing hit-and-miss, search for 7 consecutive crosses in a row, and then search for 7 consecutive crosses in a column.
    Try to find in this fashion the cross at the top left corner of the board and the delimiter of a square in the chess board. 
    '''

    # Step 1 - do Hit-And-Miss morphological operation to find the crosses
    small_kernel_size = 3
    large_kernel_size = small_kernel_size*4
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(large_kernel_size,large_kernel_size))
    # make the center un-crossed
    kernel_cross[large_kernel_size//2 - small_kernel_size//2: large_kernel_size//2 + small_kernel_size//2+1,
                  large_kernel_size//2 - small_kernel_size//2: large_kernel_size//2 + small_kernel_size//2+1] = 0 
    img_hms = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_cross)
    original_image = draw_on_image(img_hms, original_image, (0, 255, 0)) # green

    # Step 2 - in the hms image, search for aligned crosses.
    crosses = [Cross(i,j) for i,j in np.argwhere(img_hms)]
    crosses_by_row = {i: [] for i in range(img_hms.shape[0])}
    crosses_by_col = {j: [] for j in range(img_hms.shape[1])}
    for c in crosses:
        
        # they should be sorted
        crosses_by_row[c.row].append(c)
        crosses_by_col[c.col].append(c)

    # for every cross, search for a cross in the same row and column.
    # chess is 8x8
    # i should have 7x7 of crosses in the board.
    found = False
    suspected_square_delim = 0
    top_cross = None
    for row in sorted(crosses_by_row.keys()):
        r = 1 # search in r rows below and r rows above, to allow some mistakes
        dist_mistake = 1 # allow mistake in distance between crosses
        corsses_in_row = crosses_by_row[row]
        for i in range(1, r+1):
            if row-i in crosses_by_row:
                corsses_in_row += crosses_by_row[row-i]
            if row+i in crosses_by_row:
                corsses_in_row += crosses_by_row[row+i]

        
        if len(corsses_in_row) < 7:
            continue
    
        # now i need to verify that i have 7 crosses that are sapced evenly.
        # i will sort them and then check that the difference between each two is the same.
        sorted_c_in_row = sorted(corsses_in_row, key= lambda c: c.col)
        for i, suspected_square_delim in find_even_spread_dots([c.col for c in sorted_c_in_row], dist_mistake, num_dots=7):
            
            # got a suspected starting cross, now check if the colums of it match
            suspected_cross = sorted_c_in_row[i]
            suspected_col = suspected_cross.col
            print(f"[+] found a suspected starting cross - ", {suspected_cross, suspected_square_delim})
            
            corsses_in_col = crosses_by_col[suspected_col]
            for k in range(1, r+1):
                if suspected_col-k in crosses_by_col:
                    corsses_in_col += crosses_by_col[suspected_col-k]
                if suspected_col+k in crosses_by_col:
                    corsses_in_col += crosses_by_col[suspected_col+k]

            if len(corsses_in_col) < 7:
                continue
            # sort the rows in the suspected colum
            sorted_c_in_col = sorted(corsses_in_col,  key= lambda c: c.row)
            initial_pos = sorted_c_in_col.index(suspected_cross)
            rows_to_check = [c.row for c in sorted_c_in_col[initial_pos:initial_pos+7]]
            
            if verify_even_spread_dots(rows_to_check, dist_mistake, suspected_square_delim):
                # TODO: i can even verify that each row and column now have good alignment, but i think it is not needed.  
                # I can also make this more robust to try to find the board even in case of some missing crosses.
                found = True
                top_cross = suspected_cross
                break
        if found:
            break
    
    if not found:
        print("did not find a chess board")
        return None

    # TODO: figure out why i need +1, i think this is because the cross is internal or somehting, idk.
    suspected_square_delim +=1
    # i've found a chess board - let's return the bounding box (upper left, lower right) and square delimiter of the board
    top_left = (top_cross.row - suspected_square_delim, top_cross.col - suspected_square_delim)
    bottom_right = (top_cross.row + 7*suspected_square_delim, top_cross.col + 7*suspected_square_delim)
    return (top_left, bottom_right, suspected_square_delim)

def split_chess_board(image, top_left, bottom_right, delim):
    chess_board_squares = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append(image[top_left[0] + i*delim: top_left[0] + (i+1)*delim, top_left[1] + j*delim: top_left[1] + (j+1)*delim].copy())
        chess_board_squares.append(row)
    return chess_board_squares

def display_board_image(chess_board_squares, labels=None):
    fig = plt.figure(figsize=(8, 8))    
    for i in range(8):
        for j in range(8):
            ax = fig.add_subplot(8, 8, i*8+j+1)
            if labels is not None:
                ax.set_title(labels[i*8 + j])
            ax.imshow(chess_board_squares[i][j])
            ax.axis('off')
    plt.show()

def save_board_images(chess_board_squares, name):
    # save board images
    for i in range(8):
        for j in range(8):
            cv2.imwrite(f"chessboard_identifier/analyzed/{name}_{i}_{j}.png", chess_board_squares[i][j])

# # Final goal - returns an FEN string of the chess board
# def identify_board(image):
    

def get_splitted_board(image):
    (top_left, bottom_right, delim) = identify_board(image, image)
    splitted_board = split_chess_board(image, top_left, bottom_right, delim)
    # display_board_image(splitted_board)
    # save_board_images(splitted_board, "test_image")
    # print("drawing square - ", (top_left, bottom_right, delim))
    return splitted_board


def get_board_positions(image):
    board = Board()
    splitted_board = get_splitted_board(image)
    for i in range(8):
        for j in range(8):
            board.board[i][j] = splitted_board[i][j]
    return board

def main():
    print("[+] loading image")
    image = cv2.imread('chessboard_identifier/res/board_webpage.png')
    print("[+] Initializing board")
    board = BoardIdentifier(image)
    print("[+] Identifying positions")
    board.get_board_positions()
    print("[+] Done")

    # board = identify_board(image)
if __name__ == "__main__":
    main()


