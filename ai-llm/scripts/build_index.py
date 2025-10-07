#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, M;
    cin >> n >> M;

    vector<int> w(n), v(n);
    for (int i = 0; i < n; i++) {
        cin >> w[i] >> v[i];  
    }

    vector<vector<int>> dp(n + 1, vector<int>(M + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int cap = 0; cap <= M; cap++) {
            dp[i][cap] = dp[i - 1][cap];
            if (w[i - 1] <= cap) {
                dp[i][cap] = max(dp[i][cap],
                                 dp[i - 1][cap - w[i - 1]] + v[i - 1]);
            }
        }
    }

    vector<int> chosen;
    int cap = M;
    for (int i = n; i >= 1; i--) {
        if (dp[i][cap] != dp[i - 1][cap]) {
            chosen.push_back(i - 1);
            cap -= w[i - 1];
        }
    }
    reverse(chosen.begin(), chosen.end());

    for (int id : chosen) {
        cout << id << "\n"; 
    }

    return 0;
}
