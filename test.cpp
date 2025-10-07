#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n,b;
    cin>>n>>b;
    
    for(int i = 0; i < n*2 - 1; i++) {
        for(int j = 0; j < abs(n - 1 -i); j++) 
            cout<<" ";
        
        for(int j = 0;j < abs(n - 1 - i) + 1; j++)
            cout<<"*";
        
        cout<<endl;

    }
    
    
    for(int i = 0; i < n * 2 - 1; i++) {
        for(int j = 0; j < abs(n - (abs(n - 1 - i) + 1)) * 2; j++) 
            cout<<" ";
         
        for(int j = 0;j < abs(n - 1 - i) + 1; j++)
            cout<<"*";
        
        cout<<endl;

    }
    return;
}

int main() {
    solve();
    return 0;
}
