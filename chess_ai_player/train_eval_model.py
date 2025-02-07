"""
Finish all TODO's
Train loss also doesn't decreases much :( check if there is a problem there.
TODO: Fix bot.py to know if your are white or black.
"""
import datetime
import time
import os
import logging
import random
import sys
import multiprocessing as mp


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor

BITBOARD_SIZE = 773
HIDDEN_LAYER_SIZE = 1024
DROPOUT_COUNT = 0.1
BATCH_SIZE = 128
TESTSET_SIZE = 2500
EPOCH_SAVE_INTERVAL = 3 # TODO: Find something better than this mechanism
EPOCHS_COUNT = 100
GOOD_EVAL_DIFF = 2

IN_GOOGLE_COLAB = True if os.getenv("COLAB_RELEASE_TAG") else False

if IN_GOOGLE_COLAB:
    DATASET_SIZE = "12M"
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/chessData100K.csv"
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/merged_big_fixed_dataset.csv"
    TEST_EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/short_tactics_test_300K.csv"
    FILEPATH = "/content/drive/My Drive/Colab Notebooks"
    CPU_CORES_COUNT = os.cpu_count()
else:
    EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\short_tactics_test_10K.csv'
    TEST_EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\short_tactics_test_10K.csv'
    DATASET_SIZE = 100_000
    FILEPATH = ""
    CPU_CORES_COUNT = 1

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def fen_to_bitboard(fen: str):
    """Converts a FEN string to a board binary representation.
    FEN example - N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R b KQ - 0 12
    N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R         b                  KQ                   			-			 		     0        				 12
    --------------- pieces positions ---------------------player turn----castling rights---------possible en passsent targets--------half move clock-------full move number-------------
    """
    empty = 0
    white_pawns = 0
    white_knights = 0
    white_bishops = 0
    white_rooks = 0
    white_queens = 0
    white_kings = 0
    black_pawns = 0
    black_knights = 0
    black_bishops = 0
    black_rooks = 0
    black_queens = 0
    black_kings = 0

    fen_board_part, fen_turn_part, fen_castling_part, *_ = fen.split()

    # Translate FEN board representation to bitboards

    # Start from the 8th rank and a-file (0-based index)
    row = 7  
    file = 0 

    for char in fen_board_part:
        if char == '/':
			# Continue to next line
            row -= 1
            file = 0
        elif char.isnumeric():
        	# Numeric means empty spaces, so we skip accordingly to the next piece in row.
            file += int(char)
        else:
        	# Convert rank and file to a square index
            square = row * 8 + file 

            if char == 'P':
                white_pawns |= 1 << square
            elif char == 'N':
                white_knights |= 1 << square
            elif char == 'B':
                white_bishops |= 1 << square
            elif char == 'R':
                white_rooks |= 1 << square
            elif char == 'Q':
                white_queens |= 1 << square
            elif char == 'K':
                white_kings |= 1 << square
            elif char == 'p':
                black_pawns |= 1 << square
            elif char == 'n':
                black_knights |= 1 << square
            elif char == 'b':
                black_bishops |= 1 << square
            elif char == 'r':
                black_rooks |= 1 << square
            elif char == 'q':
                black_queens |= 1 << square
            elif char == 'k':
                black_kings |= 1 << square

            file += 1

    pieces = [
    	white_pawns, white_knights, white_bishops, white_rooks, white_queens, white_kings, 
    	black_pawns, black_knights, black_bishops, black_rooks, black_queens, black_kings
    ]
    board_pieces_bits = []
    for piece in pieces:
    	# Pad to 64 bits for each piece and skip the '0b' prefix
    	piece_bits_str = bin(piece)[2:].zfill(64)
    	piece_bits = [int(bit) for bit in piece_bits_str]

    	board_pieces_bits.extend(piece_bits)

    # Determine player turn
    turn_bits = [1 if fen_turn_part == 'w' else 0]

    # Determine castling rights
    in_castling = lambda x: 1 if x in fen_castling_part else 0
    white_kingside_castle = in_castling('K')
    white_queenside_castle = in_castling('Q')
    black_kingside_castle = in_castling('k')
    black_queenside_castle = in_castling('q')

    castling_bits = [white_kingside_castle, white_queenside_castle, black_kingside_castle, black_queenside_castle]

    full_board_bits = board_pieces_bits + turn_bits + castling_bits
    board_array = np.array(full_board_bits, dtype=np.bool_)

    return board_array

def is_accurate_prediction(score: float, prediction: float) -> bool:
    diff = abs(score-prediction)
    if diff < GOOD_EVAL_DIFF:
        return True
    
    return False


def _process_chunk(chunk, shared_list):
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logging.info(f"Starting to load chunk, starting loading on {timestamp}, chunk index: {chunk.index}")

    all_fens = [] 
    all_evals = [] 
    for data in chunk.iloc:
        try:
            fen = data['FEN']
            bitboard = fen_to_bitboard(fen)
            eval_score = int(data['Evaluation'])
        except ValueError as e:
            logging.warning(f"Found problematic item in dataset: fen - {fen}, score - {eval_score}. Skipping")
            continue

        all_fens.append(bitboard)
        all_evals.append(eval_score)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logging.info(f"Finished to load chunk, finished loading on {timestamp}, chunk index: {chunk.index}")

    shared_list.append((all_fens, all_evals))
    logging.debug(f"Returning from multi process _process_chunk")
    return 

class ChessEvaluationsDataset(Dataset):
    DEFAULT_ITEM = ("r1bqk2r/pppp1ppp/2n1pn2/3P4/1b2P3/2N2Q2/PPP2PPP/R1B1KBNR b KQkq - 3 5", -0.6)

    def __init__(self, evaluations_file: str, test_data_mode: bool = False):
        csv_iter = pd.read_csv(evaluations_file, chunksize=10_000)

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Starting to load dataset. Will load all to GPU, starting loading on {timestamp}")
        parsed_data = []

        with mp.Manager() as manager:
            processes = []
            shared_list = manager.list()
            for i, chunk in enumerate(csv_iter):
                if len(processes) >= CPU_CORES_COUNT:
                    processes.pop(0).join()
                
                train_chunk, test_chunk = train_test_split(chunk, test_size=0.1, random_state=15)
                chunk = test_chunk if test_data_mode else train_chunk

                p = mp.Process(target=_process_chunk, args=(chunk, shared_list))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            parsed_data = list(shared_list)

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Finished loading all dataset on {timestamp}")

        fens_arrays = [data[0] for data in parsed_data]
        evals_arrays = [data[1] for data in parsed_data]
        fens_data = np.concatenate(fens_arrays, axis=0)
        evals_data = np.concatenate(evals_arrays, axis=0)
        self.fens_tensor = torch.from_numpy(fens_data).to(DEVICE, dtype=torch.bool)
        self.evals_tensor = torch.from_numpy(evals_data).to(DEVICE, dtype=torch.int8).view(-1, 1)
        assert len(self.fens_tensor) == len(self.evals_tensor), "Unequal fens and evaluations, unexpected and should be debugged"

        logging.info(f"Finished parsing data!")

    def __len__(self):
        return len(self.fens_tensor)

    def __getitem__(self, index: int):
        return self.fens_tensor[index].to(torch.float32), self.evals_tensor[index].to(torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=BITBOARD_SIZE, out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_COUNT),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_COUNT),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=1),
        )

    def forward(self, x):
        """The forward passing of an element through the network"""
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_start_time = time.time()
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    train_loss = 0
    batches_tested = 0
    for batch, (board_positions, real_evaluations) in enumerate(dataloader):
        # Compute prediction and loss
        predictions = model(board_positions)
        loss = loss_fn(predictions, real_evaluations)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(board_positions)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        train_loss += loss
        batches_tested += 1

    logging.info(f"Train Error: \n Avg loss: {train_loss/batches_tested:>8f} \n Train epoch took: {time.time() - train_start_time} seconds")
    

def manual_test(model):
    model.eval()
    board_to_eval = {
        'r1b1k2r/pppp1ppp/2n2q2/8/4P3/N4N2/PPP2PPP/R2QKB1R w KQkq - 0 8': 8.6,
        '3r4/5k1p/2p3p1/1p1q4/1P2p3/Q3P2P/3p1PP1/3R2K1 b - - 9 43': -3.6
    }
    test_results = []
    for board, eval in board_to_eval.items():
        board_tensor = torch.from_numpy(fen_to_bitboard(board)).to(DEVICE, dtype=torch.float32)
        result = model(board_tensor).item()
        logging.info(f"Test result: {result}, should be: {eval}")
        result_diff = abs(result-eval)
        test_results.append(result_diff)
    
    avg_result = sum(test_results)/len(test_results)
    return avg_result

def test_loop(dataloader, model, loss_fn):
    test_start_time = time.time()
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    test_loss = 0
    samples_tested = 0
    batches_tested = 0
    in_range = 0
    #stopping_after = 200_000

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for chess_boards, boards_eval in dataloader:
            predictions = model(chess_boards)
            test_loss += loss_fn(predictions, boards_eval).item()

            for board_eval, prediction in zip(boards_eval, predictions):
                if is_accurate_prediction(int(board_eval), int(prediction)):
                    in_range += 1


            batches_tested += 1
            samples_tested = batches_tested * dataloader.batch_size
            #if samples_tested >= 1_000:
            #	break

    logging.info(f"Test Error: \n Avg loss: {test_loss/batches_tested:>8f} \n Test took: {time.time() - test_start_time} seconds")
    logging.info(f"Good evals (by range) percentage: {in_range/samples_tested * 100:>5f}")

    if test_loss < 3:
        logging.info(f"Found model with avg loss of: {test_loss}, saving it.")
        save_model(model)

def train_network(model, train_dataloader, test_dataloader):
    loss_fn = nn.MSELoss()
	# learning_rate = 1e-3
    learning_rate = 0.01
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epochs = EPOCHS_COUNT
    for t in range(epochs):
        logging.info
        logging.info(f"------------------------------- Starting epoch {t+1} -------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        manual_test(model)
        test_loop(test_dataloader, model, loss_fn)
        if t % EPOCH_SAVE_INTERVAL == 0:
            save_model(model)
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.flush()


    logging.info("Done training!")


def save_model(model):
    logging.info("Saving model")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    test_result = int(manual_test(model))
    filename = os.path.join(FILEPATH, f"model-T-{test_result}-E-{EPOCHS_COUNT}-D-{DATASET_SIZE}-R-{random.randint(1, 99)}-{timestamp}.pth")
    if IN_GOOGLE_COLAB:
        torch.save(model, "/content/drive/My Drive/model_2_mil.pth")
    torch.save(model, filename)


def continue_train(model, train_data, test_data):
	train_network(model, train_data, test_data)

def setup_logging():
    console_handler = logging.StreamHandler()
    log_path = os.path.join(FILEPATH,'ML_chess_training.log') if IN_GOOGLE_COLAB else 'log.txt'
    print(f"Logging in {log_path}")
    file_handler = logging.FileHandler(log_path)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(processName)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    #logger = logging.getLogger('ML_trainer_logger')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info("Logger is configured")

    # Log some messages
    logger.info("This message appears in both the console and the log file.")
    logger.debug("This debug message will only appear in the log file.")

if __name__ == "__main__":
    if IN_GOOGLE_COLAB: 
        print("Running in google colab")
    else:
        print("Running outside of google colab")
    print(f"Using {DEVICE} device")
    print(f"Number of available CPU cores: {CPU_CORES_COUNT}")
    setup_logging()
    chess_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH)
    train_dataloader = DataLoader(chess_dataset, batch_size=BATCH_SIZE, shuffle=True)
    chess_test_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH, test_data_mode=True)
    test_dataloader = DataLoader(chess_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = NeuralNetwork().to(DEVICE)
    train_network(model, train_dataloader, test_dataloader)
    save_model(model)
