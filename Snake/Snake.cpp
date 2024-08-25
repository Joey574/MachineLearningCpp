#include <iostream>
#include <random>

#include "Matrix.h"

class Snake {
public:
	struct pos {
		int x;
		int y;
	};

	Snake(int width, int height) {
		gameboard = Matrix(height, width, 0.0f);

		spawn_snake();
		spawn_food();
	}

	/// <summary>
	/// 1 -> up
	/// 2 -> down
	/// 3 -> left
	/// 4 -> right
	/// </summary>
	/// <param name="direction"></param>
	/// <returns></returns>
	bool move_snake(int direction) {

		// Used if collision w food
		pos temp = snake_pos.back();

		// Update locations
		for (int i = snake_pos.size() - 1; i > 0; i--) {
			snake_pos[i] = snake_pos[i - 1];
		}

		switch (direction) {
		case 1:
			snake_pos[0].y--;
			break;
		case 2:
			snake_pos[0].y++;
			break;
		case 3:
			snake_pos[0].x--;
			break;
		case 4:
			snake_pos[0].x++;
			break;
		}

		// Check if collision w snake or out of bounds
		if (!in_bounds(snake_pos[0])) {
			return false;
		}

		// Check if collision w food
		if (gameboard(snake_pos[0].y, snake_pos[0].x) == 1) {
			snake_pos.push_back(temp);
			gameboard(temp.y, temp.x) = 2;
			spawn_food();
		} else {
			gameboard(temp.y, temp.x) = 0;
		}

		// Update head location on gameboard
		gameboard(snake_pos[0].y, snake_pos[0].x) = 2;
	}

	void draw_gameboard() {
		std::cout << "\u001b[H";
		for (int r = 0; r < gameboard.RowCount; r++) {
			for (int c = 0; c < gameboard.ColumnCount; c++) {
				switch ((int)gameboard(r, c)) {
				case 1:
					std::cout << "X ";
					break;
				case 2:
					std::cout << "O ";
					break;
				default:
					std::cout << "_ ";
				}
			}
			std::cout << "\n";
		}
	}

	void reset(int width, int height) {
		gameboard = Matrix(height, width, 0.0f);

		snake_pos.clear();
		spawn_snake();
		spawn_food();
	}

	Matrix gameboard;
	std::vector<pos> snake_pos;
private:

	void spawn_snake() {
		int r_mid = gameboard.RowCount / 2;
		int c_mid = gameboard.ColumnCount / 2;

		gameboard(r_mid, c_mid - 1) = 2;
		gameboard(r_mid, c_mid) = 2;
		gameboard(r_mid, c_mid + 1) = 2;

		snake_pos = std::vector<pos>(3);
		snake_pos[0] = { c_mid - 1, r_mid };
		snake_pos[1] = { c_mid, r_mid };
		snake_pos[2] = { c_mid + 1, r_mid };
	}

	void spawn_food() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> row(0, gameboard.RowCount);
		std::uniform_int_distribution<> col(0, gameboard.ColumnCount);

		int r = row(rd);
		int c = col(rd);

		while (gameboard(r, c) != 0) {
			r = row(rd);
			c = col(rd);
		}

		gameboard(r, c) = 1;
	}

	inline bool in_bounds(pos p) {
		return p.x > -1 && p.y > -1 && p.x < gameboard.ColumnCount && p.y < gameboard.RowCount && gameboard(p.y, p.x) != 2;
	}
};