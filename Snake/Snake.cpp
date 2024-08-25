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
		switch (direction) {
		case 1:

			break;
		case 2:

			break;
		case 3:

			break;
		case 4:

			break;
		}
	}

	void draw_gameboard() {
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
		snake_pos[0] = { r_mid, c_mid - 1};
		snake_pos[1] = { r_mid, c_mid };
		snake_pos[2] = { r_mid, c_mid + 1};
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

	void collision_checks() {

	}

};