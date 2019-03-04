#!/usr/bin/env python3

import os
import chess.pgn
import numpy as np

def serialize(board):
	state = np.zeros(64, np.uint8)
	for i in range(64):
		current_piece = board.piece_at(i)
		if (current_piece is not None):
			state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
						"p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[current_piece.symbol()]
		else:
			state[i] = 0
	state = state.reshape(8,8)
	binary_state = np.zeros((5,8,8), np.uint8)
	binary_state[0] = (state>>3)&1
	binary_state[1] = (state>>2)&1
	binary_state[2] = (state>>1)&1
	binary_state[3] = (state>>0)&1

	binary_state[4] = board.turn

	return state

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

			game = chess.pgn.read_game(pgn)
			if (game):
				games += 1
				board = game.board()
				for move in game.mainline_moves():
					result = RESULT_VALUES[game.headers['Result']]
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

X,Y = build_dataset(2000000)
np.savez("processed/dataset_2M.npz", X, Y)
