import chess
import torch

from train_eval_model import *


def find_best_move(board_fen: str, eval_model):
	board = chess.Board(board_fen)
	best_move = None
	best_eval = -100
	print(f"Current board:\n{board}")
	for move in board.legal_moves:
		board.push(move)
		possible_fen = board.fen()

		bitboard = fen_to_bitboard(possible_fen)
		bitboard_tensor = torch.from_numpy(bitboard).to(DEVICE, dtype=torch.float32)
		new_eval = eval_model(bitboard_tensor)
		print(f"Move: {move} give score of: {float(new_eval)}")
		if new_eval > best_eval:
			best_move = move
			best_eval = new_eval

		board.pop()

	print(f"Best move is: {best_move}, and it gives evaluation of: {best_eval}")
	return best_move


if __name__ == "__main__":
	board_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq - 2 4"
	eval_model = torch.load('res/model.pth')
	eval_model.eval() 
	find_best_move(board_fen, eval_model)