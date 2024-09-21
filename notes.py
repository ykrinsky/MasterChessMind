"""
TODO:
* Think on how to normalize the evaluations and how to create the output layer !
* Train the network.
* Deal with evaluation with #

"""

# Input layer - FEN to bitboard
# FEN example - N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R b KQ - 0 12
# N2k1bnr/3p1ppp/b1n1p3/8/4P3/P4N2/1PP2PPP/R1B1K2R         b                  KQ                   			-			 		     0        				 12
# --------------- pieces positions ---------------------player turn----castling rights---------possible en passsent targets--------half move clock-------full move number-------------
import datetime
import time

import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor

BITBOARD_SIZE = 773
CENTIPAWN_UNIT = 100
MAX_LEAD_SCORE = 15

EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\tactic_evals.csv'
DATASET_SIZE = 1_250_000
EPOCHS_COUNT = 100

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

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
    full_board_tensor = torch.tensor(full_board_bits, dtype=torch.float)

    return full_board_tensor


class ChessEvaluationsDataset(Dataset):
    def __init__(self, evaluations_file: str):
        all_evaluations = pd.read_csv(evaluations_file)
        # TODO: Deal with ending positions and specifically mates (#)
        relevant_evaluations = all_evaluations.loc[~all_evaluations['Evaluation'].str.contains('#')]
        relevant_evaluations = relevant_evaluations[:DATASET_SIZE]
        self.evaluations = relevant_evaluations

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, index: int):
    	fen = self.evaluations.iloc[index]['FEN']
    	raw_eval_score = self.evaluations.iloc[index]['Evaluation']
    	eval_score = int(raw_eval_score) / CENTIPAWN_UNIT
    	eval_score = max(eval_score, -MAX_LEAD_SCORE)
    	eval_score = min(eval_score, MAX_LEAD_SCORE)
    	# eval_score = eval_score / 10

    	eval_tensor = torch.tensor([eval_score]).to(device)

    	bitboard_tensor = fen_to_bitboard_tensor(fen).to(device)
    	# print(f"Got item from index: {index}, eval: {eval_tensor}, bitboard: {bitboard_tensor}")

    	return bitboard_tensor, eval_tensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=BITBOARD_SIZE, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(board_positions)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def bad_test_loop(model):
	test_board = fen_to_bitboard_tensor('r1b1k2r/pppp1ppp/2n2q2/8/4P3/N4N2/PPP2PPP/R2QKB1R w KQkq - 0 8').to(device)
	test_result = model(test_board)
	print(f"Test result: {test_result}, should be: +8.6")
	return test_result

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    test_start_time = time.time()
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    batches_tested = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            batches_tested += 1
            if batches_tested >= 500:
            	break

    test_loss /= batches_tested
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n Test took: {time.time() - test_start_time} seconds")

def print_model_param(model):
	print(f"Model structure: {model}\n\n")

	for name, param in model.named_parameters():
    		print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

def train_network(model, train_dataloader, test_dataloader):
	loss_fn = nn.MSELoss()
	# learning_rate = 1e-3
	learning_rate = 0.01
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	epochs = EPOCHS_COUNT
	for t in range(epochs):
	    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	    print(f"Epoch {t+1} on {timestamp}\n-------------------------------")
	    train_loop(train_dataloader, model, loss_fn, optimizer)
	    bad_test_loop(model)
	    test_loop(test_dataloader, model, loss_fn)

	print("Done!")

def save_model(model):
	# TODO: Save model name should be include hyper parametes (learning rate, epochs, etc.)
	timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	test_result = int(bad_test_loop(model))
	filename = f"model-T-{test_result}-E-{EPOCHS_COUNT}-D-{DATASET_SIZE}-{timestamp}.pth"
	torch.save(model, filename)


def continue_train(model, train_data, test_data):
	train_network(model, train_data, test_data)


def main(train_data, test_data):
	train_network(model, train_data, test_data)
	save_model(model)

chess_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH)
train_dataloader = DataLoader(chess_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(chess_dataset, batch_size=64, shuffle=True)
model = NeuralNetwork().to(device)
