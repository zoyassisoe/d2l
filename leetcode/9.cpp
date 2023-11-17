#include <iostream>
#include <string>

using namespace std;
bool isPalindrome(int x)
{
    // 123
    if (x < 0)
        return false;
    string s;
    while (x)
    {
        s.push_back(x % 10 + '0');
        x /= 10;
    }
    int n = s.length();
    for (int i = 0; i < n / 2; i++)
    {
        if (s[i] != s[n - i - 1])
            return false;
    }
    return true;
}

int main()
{
    string s;
    cout << isPalindrome(1001);

    return 0;
}

class Solution
{
public:
    bool isPalindrome(int x)
    {
        // 123
        if (x < 0)
            return false;
        string s;
        while (x)
        {
            s.push_back(x % 10 + '0');
            x /= 10;
        }
        int n = s.length();
        for (int i = 0; i < n / 2; i++)
        {
            if (s[i] != s[n - i - 1])
                return false;
        }
        return true;
    }
};