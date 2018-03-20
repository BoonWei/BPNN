# PAT-BASIC
#include <iostream>
#include <string>
using namespace std;

int main()
{
	string s;
	cin >> s;
	int len = s.length();
	if (s[0] == '-') {
		cout << '-';
	}

	int E = 0;
	for (int i = 0; i < len; i++) {
		if (s[i] == 'E') {
			E = i;
			break;
		}
	}
	int sum = 0;
	for (int i = E + 2; i < len; i++) {
		sum = sum * 10;
		sum = sum + (s[i] - '0');
	}
	if (s[E + 1] == '-') {
		if (sum > 0) {
			cout << "0.";
			for (int i = 1; i < sum; i++) {
				cout << '0';
			}
			for (int i = 1; i < E; i++) {
				if (s[i] != '.') {
					cout << s[i];
				}
			}
		}else{
			for (int i = 1; i < E; i++) {
				cout << s[i];
			}
		}
	}
	else {
		if (E - 3 < sum) {
			if (s[1] != 0) {
				cout << s[1];
			}
			for (int i = 3; i < E; i++) {
				if (s[i] != '.') {
					cout << s[i];
				}
			}
			for (int i = 1; i < sum; i++) {
				cout << '0';
			}
		}else{
			if (s[1] != 0) {
				cout << s[1];
			}
			for (int i = 3; i < E; i++) {
				if (sum + 3 == i) {
					cout << '.';
				}
				if (s[i] != '.') {
					cout << s[i];
				}
			}
		}
	}

    return 0;
}
