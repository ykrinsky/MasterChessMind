"""
TODO:
* Think on how to normalize the evaluations and how to create the output layer !
* Train the network.
* Deal with evaluation with #
* Think on different neural network you can do with chess.
* Add minimax algorithm to move calculation.
* See what is workers and if it can speed up the training
* DONE - look better at loss function (== learned that it the mean squared error, and loss of 3 for example is pretty good, as it means ~1.7 error)
* maybe add option to play against the bot

download the model
early stopping function (when loss is under 1 for example)
read again the researches
learn how to streamline data better to the GPU
Add logging or pring loss of training set over time.
"""

# Input layer - FEN to bitboard
# FEN example - N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R b KQ - 0 12
# N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R         b                  KQ                   			-			 		     0        				 12
# --------------- pieces positions ---------------------player turn----castling rights---------possible en passsent targets--------half move clock-------full move number-------------
import datetime
import time
import os
import logging
import random

import torch
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor

BITBOARD_SIZE = 773
CENTIPAWN_UNIT = 100
MAX_LEAD_SCORE = 15
HIDDEN_LAYER_SIZE = 512
BATCH_SIZE = 64
TESTSET_SIZE = 2500

IN_GOOGLE_COLAB = True if os.getenv("COLAB_RELEASE_TAG") else False

if IN_GOOGLE_COLAB:
    print("Running in google colab")
    # EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/tactic_evals_short.csv"
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/tactic_evals.csv"
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/chessDataShort.csv"
    DATASET_SIZE = 4_500_000
    FILEPATH = "/content/drive/My Drive/Colab Notebooks"
else:
    print("NOT in google colab")
    EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\tactic_evals.csv'
    DATASET_SIZE = 2_000
    FILEPATH = ""

#DATASET_SIZE = 1_250_000
EPOCHS_COUNT = 200

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

def fen_to_bitboard_tensor(fen: str):
    # Initialize bitboards for different piece types
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

    # Split FEN into parts
    fen_board_part, fen_turn_part, fen_castling_part, *_ = fen.split()

    # Translate FEN board representation to bitboards

    # Start from the 8th rank and a-file (0-based index)
    row = 7  
    col = 0 

    for char in fen_board_part:
        if char == '/':
			# Continue to next line
            row -= 1
            col = 0
        elif char.isnumeric():
        	# Numeric means empty spaces, so we skip accordingly to the next piece in row.
            col += int(char)
        else:
        	# Convert rank and file to a square index
            square = row * 8 + col 

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

            col += 1

    pieces = [
    	white_pawns, white_knights, white_bishops, white_rooks, white_queens, white_kings, 
    	black_pawns, black_knights, black_bishops, black_rooks, black_queens, black_kings
    ]
    pieces_board_bits = []
    for piece in pieces:
    	# Pad to 64 bits for each piece and skip the '0b' prefix
    	piece_bits_str = bin(piece)[2:].zfill(64)
    	piece_bits = [int(bit) for bit in piece_bits_str]

    	pieces_board_bits.extend(piece_bits)

    # Determine player turn
    turn_bits = [1 if fen_turn_part == 'w' else 0]

    # Determine castling rights
    in_castling = lambda x: 1 if x in fen_castling_part else 0
    white_kingside_castle = in_castling('K')
    white_queenside_castle = in_castling('Q')
    black_kingside_castle = in_castling('k')
    black_queenside_castle = in_castling('q')

    castling_bits = [white_kingside_castle, white_queenside_castle, black_kingside_castle, black_queenside_castle]

    # Calculate occupancy board (and it's investion to get emptry squares) - NOT SURE IS NEEDED
    # occupancy = pawns | knights | bishops | rooks | queens | kings
    # empty = ~occupancy & 0xFFFFFFFFFFFFFFFF

    full_board_bits = pieces_board_bits + turn_bits + castling_bits
    full_board_tensor = torch.tensor(full_board_bits, dtype=torch.float32)
    #import IPython;IPython.embed()

    return full_board_tensor


class ChessEvaluationsDataset(Dataset):
    def __init__(self, evaluations_file: str, is_for_test = False):
        all_evaluations = pd.read_csv(evaluations_file)
        # TODO: Deal with ending positions and specifically mates (#)
        relevant_evaluations = all_evaluations.loc[~all_evaluations['Evaluation'].astype(str).str.contains('#')]
        if is_for_test:
            relevant_evaluations = relevant_evaluations[::-1][:TESTSET_SIZE]
        else:
            relevant_evaluations = relevant_evaluations[:DATASET_SIZE]

        self.evaluations = relevant_evaluations
        all_fens = [] 
        all_evals = [] 
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Starting to load dataset. Will load all to GPU, starting loading on {timestamp}")
        # for fen
        for index, data in enumerate(relevant_evaluations.iloc):
            if index % 200_000 == 0:
                logging.debug(f"Loaded until now, {index} positions")
            fen = data['FEN']
            #bitboard = fen_to_bitboard_tensor(fen).tolist()
            bitboard = fen_to_bitboard_tensor(fen).to(dtype=torch.bool)
            all_fens.append(bitboard)

            raw_eval_score = data['Evaluation']
            eval_score = int(raw_eval_score) / CENTIPAWN_UNIT
            eval_score = max(eval_score, -MAX_LEAD_SCORE)
            eval_score = min(eval_score, MAX_LEAD_SCORE)
            eval_tensor = torch.tensor([eval_score], dtype=torch.int8, device=DEVICE)
            all_evals.append(eval_tensor)

        self.fens_tensor = torch.stack(all_fens).to(DEVICE)
        self.evals_tensor = torch.stack(all_evals).to(DEVICE)

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Finished loading all dataset!! Finished at {timestamp}")

        #import IPython; IPython.embed()


    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, index: int):
        return self.fens_tensor[index].to(torch.float32), self.evals_tensor[index].to(torch.float32)
        """
        fen = self.evaluations.iloc[index]['FEN']
    	raw_eval_score = self.evaluations.iloc[index]['Evaluation']
    	eval_score = int(raw_eval_score) / CENTIPAWN_UNIT
    	eval_score = max(eval_score, -MAX_LEAD_SCORE)
    	eval_score = min(eval_score, MAX_LEAD_SCORE)
    	# eval_score = eval_score / 10

    	eval_tensor = torch.tensor([eval_score]).to(DEVICE)

    	bitboard_tensor = fen_to_bitboard_tensor(fen).to(DEVICE)
    	# print(f"Got item from index: {index}, eval: {eval_tensor}, bitboard: {bitboard_tensor}")

    	return bitboard_tensor, eval_tensor
        """


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=BITBOARD_SIZE, out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=1),
        )

    def forward(self, x):
        """The forward passing of an element through the network"""
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (board_positions, real_evaluations) in enumerate(dataloader):
        # Compute prediction and loss
        #import pdb;pdb.set_trace()
        predictions = model(board_positions)
        loss = loss_fn(predictions, real_evaluations)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(board_positions)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def manual_test(model):
    board_to_eval = {
        'r1b1k2r/pppp1ppp/2n2q2/8/4P3/N4N2/PPP2PPP/R2QKB1R w KQkq - 0 8': 8.6,
        '3r4/5k1p/2p3p1/1p1q4/1P2p3/Q3P2P/3p1PP1/3R2K1 b - - 9 43': -3.6
    }
    test_results = []
    for board, eval in board_to_eval.items():
        board_tensor = fen_to_bitboard_tensor(board).to(DEVICE)
        result = model(board_tensor).item()
        logging.info(f"Test result: {result}, should be: {eval}")
        result_diff = abs(result-eval)
        test_results.append(result_diff)
    
    avg_result = sum(test_results)/len(test_results)
    return avg_result

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    test_start_time = time.time()
    model.eval()
    test_avg_loss = 0
    samples_tested = 0
    batches_tested = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for chess_boards, boards_evals in dataloader:
            pred = model(chess_boards)
            test_avg_loss += loss_fn(pred, boards_evals).item()
            batches_tested += 1
            samples_tested = batches_tested * dataloader.batch_size
            if samples_tested >= 1_000:
            	break

    test_avg_loss /= batches_tested
    logging.info(f"Test Error: \n Avg loss: {test_avg_loss:>8f} \n Test took: {time.time() - test_start_time} seconds")

    if test_avg_loss < 3:
        logging.info(f"Found model with avg loss of: {test_avg_loss}, saving it.")
        save_model(model)

def print_model_param(model):
	print(f"Model structure: {model}\n\n")

	for name, param in model.named_parameters():
    		print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

def train_network(model, train_dataloader, test_dataloader):
    loss_fn = nn.MSELoss()
	# learning_rate = 1e-3
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = EPOCHS_COUNT
    for t in range(epochs):
        logging.info
        logging.info(f"------------------------------- Starting epoch {t+1} -------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        manual_test(model)
        test_loop(test_dataloader, model, loss_fn)
        if t % 10 == 0:
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
    log_path = os.path.join(FILEPATH,'ML_chess_training.log')
    print(f"Logging in {log_path}")
    log_path2 = "/content/drive/My Drive/ML_chess_training_2.log" if IN_GOOGLE_COLAB else 'log.txt'
    print(f"Logging in {log_path2}")
    file_handler = logging.FileHandler(log_path)
    file_handler2 = logging.FileHandler(log_path2)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    file_handler2.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    file_handler2.setFormatter(formatter)

    #logger = logging.getLogger('ML_trainer_logger')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(file_handler2)

    logging.info("Logger is configured")

    # Log some messages
    logger.info("This message appears in both the console and the log file.")
    logger.debug("This debug message will only appear in the log file.")

if __name__ == "__main__":
    setup_logging()
    chess_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH)
    train_dataloader = DataLoader(chess_dataset, batch_size=BATCH_SIZE, shuffle=True)
    chess_test_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH, is_for_test=True)
    test_dataloader = DataLoader(chess_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = NeuralNetwork().to(DEVICE)
    train_network(model, train_dataloader, test_dataloader)
    save_model(model)
