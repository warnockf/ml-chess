#!/usr/bin/env python3

import os
import chess.pgn
import chess
import numpy as np

def serialize(board):
	assert board.is_valid()
	state = np.zeros(64, np.uint8)
	for i in range(64):
		current_piece = board.piece_at(i)
		if (current_piece is not None):
			state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
						"p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[current_piece.symbol()]

	if board.has_queenside_castling_rights(chess.WHITE):
		assert state[0] == 4
		state[0] = 7
	if board.has_kingside_castling_rights(chess.WHITE):
		assert state[7] == 4
		state[7] = 7
	if board.has_queenside_castling_rights(chess.BLACK):
		assert state[56] == 8+4
		state[56] = 8+7
	if board.has_kingside_castling_rights(chess.BLACK):
		assert state[63] == 8+4
		state[63] = 8+7

	if board.ep_square is not None:
		assert state[board.ep_square] == 0
		state[board.ep_square] = 8

	state = state.reshape(8,8)
	binary_state = np.zeros((5,8,8), np.uint8)
	binary_state[0] = (state>>3)&1
	binary_state[1] = (state>>2)&1
	binary_state[2] = (state>>1)&1
	binary_state[3] = (state>>0)&1

	binary_state[4] = board.turn

	return binary_state

def build_dataset(samples):
	game = None
    
	X = []
	Y = []

	RESULT_VALUES = { '1-0':1, '1/2-1/2':0, '0-1':-1 }

	files, moves, games = 0,0,0

	for file in os.listdir('training_sets'):
		pgn = open(os.path.join('training_sets', file))
		files += 1
		while True:
			try:
				game = chess.pgn.read_game(pgn)
			except:
				continue

			if game:
				games += 1
				board = game.board()
				for move in game.mainline_moves():
					if game.headers['Result'] in RESULT_VALUES:
						result = RESULT_VALUES[game.headers['Result']]
					else:
						continue
					Y.append(result)
					board.push(move)
					ser = serialize(board)
					X.append(ser)
					moves += 1
				print('Processed %d moves in %d games in %d files' % (moves, games, files))
				if moves > samples:
					return X,Y
			else:
				break

if __name__ == "__main__": 
	X,Y = build_dataset(10000000)
	np.savez("processed/dataset_10M.npz", X, Y)
