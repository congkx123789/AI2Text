#include <bits/stdc++.h>
using namespace std;



/*** -------------------- Utils / RNG -------------------- ***/
struct RNG {
    mt19937_64 gen;
    uniform_real_distribution<double> U{0.0, 1.0};
    RNG(uint64_t seed = chrono::high_resolution_clock::now().time_since_epoch().count()) { gen.seed(seed); }
    double rnd() { return U(gen); }
    int randint(int l, int r) { // inclusive
        uniform_int_distribution<int> D(l, r);
        return D(gen);
    }
    template<class It>
    auto choice(It first, It last) {
        int n = distance(first, last);
        int k = randint(0, n - 1);
        advance(first, k);
        return *first;
    }
} rng;

/*** -------------------- Problem Model -------------------- ***/
struct Knapsack {
    int n;                     // #items
    long long W;               // capacity
    vector<long long> w, v;    // weights, values

    Knapsack(int n=0, long long W=0): n(n), W(W), w(n), v(n) {}
};

struct Solution {
    vector<uint8_t> x; // 0/1
    long long weight = 0;
    long long value  = 0;
    double fitness   = 0.0; // can incorporate penalties if needed

    // Incremental update helpers
    inline void flip_item(const Knapsack& P, int i) {
        if (x[i]) { x[i]=0; weight -= P.w[i]; value -= P.v[i]; }
        else      { x[i]=1; weight += P.w[i]; value += P.v[i]; }
    }
};

struct Params {
    int colony_size = 600;        // #employed = #onlooker = colony_size/2 (classic ABC)
    int max_iters   = 250;        // cân bằng khi colony_size lớn
    int limit       = 40;         // trials before abandonment (will be adapted)
    double scout_boost_when_stuck = 0.15; // fraction of extra scouts when progress stalls
    int elite_k     = 3;          // top-k for potential stop-signal handling
    double stop_signal_suppression = 0.5; // reduce selection prob for near-tie elites
    int cross_period = 5;         // queen–drones crossover every T iterations
    int drones_k     = 3;         // #drones chosen for crossover
    double amplify_best = 1.6;    // waggle amplification multiplier for best
    double exploration_noise = 0.05; // prob. to flip random bit in local search
    bool use_softmax = true;      // selection scheme for onlookers
    double softmax_temp = 0.05;   // sẽ được anneal mỗi vòng
    int stagnation_patience = 30; // if no global improvement for this many iters -> stuck
    double pheromone_decay = 0.95;
    double pheromone_increase = 0.05;
};

/*** -------------------- Helper Functions -------------------- ***/
static inline void eval_solution(const Knapsack& P, Solution& s) {
    long long wsum = 0, vsum = 0;
    for (int i = 0; i < P.n; ++i) if (s.x[i]) { wsum += P.w[i]; vsum += P.v[i]; }
    s.weight = wsum;
    s.value  = vsum;
    if (wsum <= P.W) s.fitness = (double)vsum;
    else {
        long long over = wsum - P.W;
        s.fitness = (double)vsum - 1e6 * over;
    }
}

// Fast eval using current weight/value
static inline void eval_fast(const Knapsack& P, Solution& s) {
    if (s.weight <= P.W) s.fitness = (double)s.value;
    else {
        long long over = s.weight - P.W;
        s.fitness = (double)s.value - 1e6 * over;
    }
}

// Greedy repair to feasibility (drop items with low value/weight first)
static inline void repair_to_feasible(const Knapsack& P, Solution& s) {
    if (s.weight <= P.W) return;
    vector<int> idx;
    idx.reserve(P.n);
    for (int i = 0; i < P.n; ++i) if (s.x[i]) idx.push_back(i);
    sort(idx.begin(), idx.end(), [&](int a, int b){
        return (double)P.v[a]/P.w[a] < (double)P.v[b]/P.w[b];
    });
    for (int id : idx) {
        if (s.weight <= P.W) break;
        s.x[id] = 0;
        s.weight -= P.w[id];
        s.value  -= P.v[id];
    }
    eval_fast(P, s);
}

// Try a directed 1–1 swap to get feasible + not-worse value
static inline void try_swap11_feasible(const Knapsack& P, Solution& s) {
    if (s.weight <= P.W) return;
    int in_idx = -1, out_idx = -1;
    double worst = 1e100, best = -1e100;
    for (int i = 0; i < P.n; ++i) if (s.x[i]) {
        double dens = (double)P.v[i]/P.w[i];
        if (dens < worst) { worst = dens; in_idx = i; }
    }
    for (int j = 0; j < P.n; ++j) if (!s.x[j]) {
        double dens = (double)P.v[j]/P.w[j];
        if (dens > best) { best = dens; out_idx = j; }
    }
    if (in_idx != -1 && out_idx != -1) {
        long long newW = s.weight - P.w[in_idx] + P.w[out_idx];
        long long newV = s.value  - P.v[in_idx] + P.v[out_idx];
        if (newW <= P.W && newV >= s.value) {
            s.x[in_idx]=0; s.x[out_idx]=1;
            s.weight=newW; s.value=newV;
            s.fitness=(double)newV;
            return;
        }
    }
}

/*** Random init (randomized greedy) ***/
static inline Solution random_solution(const Knapsack& P) {
    Solution s;
    s.x.assign(P.n, 0);
    vector<int> order(P.n);
    iota(order.begin(), order.end(), 0);
    // bias a bit by value/weight density
    sort(order.begin(), order.end(), [&](int a, int b){
        double da = (double)P.v[a]/max(1LL,P.w[a]);
        double db = (double)P.v[b]/max(1LL,P.w[b]);
        if (fabs(da-db) < 1e-9) return a<b;
        return da>db;
    });
    // randomized selection
    long long wsum=0, vsum=0;
    for (int id : order) {
        if (rng.rnd() < 0.2) continue; // random skip to diversify
        if (wsum + P.w[id] <= P.W) {
            s.x[id] = 1;
            wsum += P.w[id]; vsum += P.v[id];
        }
    }
    s.weight = wsum; s.value = vsum;
    eval_fast(P, s);
    return s;
}

/*** Neighborhood operators (local exploitation), pheromone-guided ***/
static inline void local_perturb(const Knapsack& P,
                                 const vector<double>& pher,
                                 Solution& base) {
    Solution v = base;

    // 1) Guided bit flip: prefer adding high-pher, removing low-pher
    int flips = 1 + rng.randint(0, 2);
    for (int t = 0; t < flips; ++t) {
        int i = rng.randint(0, P.n - 1);
        double bestScore = -1e100; int bestIdx = i;
        for (int s = 0; s < 5; ++s) {
            int j = rng.randint(0, P.n - 1);
            double score = (v.x[j] ? -pher[j] : +pher[j]);
            if (score > bestScore) { bestScore = score; bestIdx = j; }
        }
        v.flip_item(P, bestIdx);
    }

    // 2) Occasional 1-in 1-out swap
    if (rng.rnd() < 0.5) {
        vector<int> in, out;
        in.reserve(P.n); out.reserve(P.n);
        for (int i = 0; i < P.n; ++i) (v.x[i] ? in : out).push_back(i);
        if (!in.empty() && !out.empty()) {
            int a = in[rng.randint(0, (int)in.size()-1)];
            int b = out[rng.randint(0, (int)out.size()-1)];
            v.flip_item(P, a); v.flip_item(P, b);
        }
    }

    // 3) Tiny noise
    if (rng.rnd() < 0.10) {
        int j = rng.randint(0, P.n - 1);
        v.flip_item(P, j);
    }

    eval_fast(P, v);
    if (v.weight > P.W) {
        // thử swap 1–1 thông minh trước
        try_swap11_feasible(P, v);
        if (v.weight > P.W) repair_to_feasible(P, v);
    }

    if (v.fitness >= base.fitness) base = v;
}

/*** Queen–drones crossover ***/
static inline Solution crossover_repair(const Knapsack& P, const Solution& A, const Solution& B) {
    Solution child; child.x.assign(P.n, 0);
    for (int i = 0; i < P.n; ++i) {
        uint8_t bit = (rng.rnd() < 0.5) ? A.x[i] : B.x[i];
        child.x[i] = bit;
    }
    // quick tally
    child.weight = 0; child.value = 0;
    for (int i = 0; i < P.n; ++i) if (child.x[i]) { child.weight+=P.w[i]; child.value+=P.v[i]; }
    eval_fast(P, child);
    if (child.weight > P.W) repair_to_feasible(P, child);
    return child;
}

/*** -------------------- ABC Solver -------------------- ***/
struct ABC {
    const Knapsack& P;
    Params par;

    int employed, onlooker;
    vector<Solution> sols;
    vector<int> trials;
    vector<double> pher; // “pheromone” memory for items
    Solution best;
    long long best_value_history = LLONG_MIN;
    int no_improve_iters = 0;

    ABC(const Knapsack& P, const Params& par): P(P), par(par) {
        employed = onlooker = par.colony_size / 2;
        sols.resize(employed);
        trials.assign(employed, 0);
        pher.assign(P.n, 0.5); // neutral
    }

    void init() {
        for (int i = 0; i < employed; ++i) sols[i] = random_solution(P);
        best = *max_element(sols.begin(), sols.end(),
                            [](const Solution& a, const Solution& b){ return a.fitness < b.fitness; });
        best_value_history = best.value;
    }

    // Probability of selection (waggle dance + stop-signal + amplify best)
    vector<double> build_selection_probs() {
        vector<double> fit(employed);
        double maxfit = -1e300;
        for (int i = 0; i < employed; ++i) { fit[i] = max(0.0, sols[i].fitness); maxfit = max(maxfit, fit[i]); }

        double eps = 1e-12;
        for (double &x : fit) if (x <= eps) x = eps;

        vector<double> p(employed, 0.0);
        if (par.use_softmax) {
            double T = max(1e-6, par.softmax_temp);
            double denom = 0.0;
            for (int i = 0; i < employed; ++i) { p[i] = exp((sols[i].fitness - maxfit) / T); denom += p[i]; }
            for (int i = 0; i < employed; ++i) p[i] /= max(denom, 1e-300);
        } else {
            double sum = 0.0;
            for (double x : fit) sum += x;
            for (int i = 0; i < employed; ++i) p[i] = fit[i] / max(sum, 1e-300);
        }

        int bi = 0;
        for (int i = 1; i < employed; ++i) if (sols[i].fitness > sols[bi].fitness) bi = i;
        p[bi] *= par.amplify_best;

        vector<int> ord(employed);
        iota(ord.begin(), ord.end(), 0);
        sort(ord.begin(), ord.end(), [&](int a, int b){ return sols[a].fitness > sols[b].fitness; });
        for (int r = 1; r < min(par.elite_k, employed); ++r) {
            int j = ord[r];
            if (fabs(sols[j].fitness - sols[bi].fitness) <= 1e-9 * max(1.0, fabs(sols[bi].fitness))) continue;
            if (sols[j].fitness >= sols[bi].fitness * 0.99) p[j] *= par.stop_signal_suppression;
        }

        double sum = accumulate(p.begin(), p.end(), 0.0);
        if (sum <= 0) {
            for (double &pi : p) pi = 1.0 / employed;
        } else {
            for (double &pi : p) pi /= sum;
        }
        return p;
    }

    int sample_index(const vector<double>& p) {
        double r = rng.rnd();
        double acc = 0.0;
        for (int i = 0; i < (int)p.size(); ++i) {
            acc += p[i];
            if (r <= acc) return i;
        }
        return (int)p.size()-1;
    }

    void employed_phase() {
        for (int i = 0; i < employed; ++i) {
            Solution before = sols[i];
            local_perturb(P, pher, sols[i]);
            if (sols[i].fitness > before.fitness) trials[i] = 0;
            else trials[i]++;
        }
    }

    void onlooker_phase() {
        vector<double> p = build_selection_probs();
        for (int t = 0; t < onlooker; ++t) {
            int j = sample_index(p);
            Solution cand = sols[j];
            local_perturb(P, pher, cand);
            if (cand.fitness > sols[j].fitness) { sols[j] = cand; trials[j] = 0; }
            else trials[j]++;
        }
    }

    void scout_phase(bool extra_boost=false) {
        for (int i = 0; i < employed; ++i) {
            if (trials[i] >= par.limit) {
                sols[i] = random_solution(P);
                trials[i] = 0;
            }
        }
        if (extra_boost) {
            int extra = max(1, (int)round(par.scout_boost_when_stuck * employed));
            for (int k = 0; k < extra; ++k) {
                int i = rng.randint(0, employed - 1);
                Solution s = random_solution(P);
                if (s.fitness > sols[i].fitness) { sols[i] = s; trials[i] = 0; }
            }
        }
    }

    void crossover_phase() {
        int bi = 0;
        for (int i = 1; i < employed; ++i) if (sols[i].fitness > sols[bi].fitness) bi = i;
        Solution queen = sols[bi];

        vector<int> idx(employed); iota(idx.begin(), idx.end(), 0);
        idx.erase(remove(idx.begin(), idx.end(), bi), idx.end());
        shuffle(idx.begin(), idx.end(), rng.gen);
        int K = min(par.drones_k, (int)idx.size());

        for (int k = 0; k < K; ++k) {
            Solution child = crossover_repair(P, queen, sols[idx[k]]);
            int wi = 0;
            for (int i = 1; i < employed; ++i) if (sols[i].fitness < sols[wi].fitness) wi = i;
            if (child.fitness > sols[wi].fitness) { sols[wi] = child; trials[wi] = 0; }
        }
    }

    void update_best_and_pheromone() {
        for (const auto& s : sols) if (s.fitness > best.fitness) best = s;

        for (double &ph : pher) ph *= par.pheromone_decay;

        vector<int> ord(employed); iota(ord.begin(), ord.end(), 0);
        sort(ord.begin(), ord.end(), [&](int a, int b){ return sols[a].fitness > sols[b].fitness; });
        int q = max(1, employed / 5);
        for (int r = 0; r < q; ++r) {
            const auto& s = sols[ord[r]];
            for (int i = 0; i < P.n; ++i)
                if (s.x[i]) pher[i] = min(1.0, pher[i] + par.pheromone_increase);
        }

        for (double &ph : pher) ph = min(1.0, max(1e-3, ph));

        if (best.value > best_value_history) {
            best_value_history = best.value;
            no_improve_iters = 0;
            par.limit = min(100, par.limit + 1);
        } else {
            no_improve_iters++;
            if (no_improve_iters % 10 == 0) par.limit = max(5, par.limit - 1);
        }
    }

    void run() {
        init();

        for (int it = 1; it <= par.max_iters; ++it) {
            // anneal softmax temperature: start explorative, end exploitative
            double t = (double)it / max(1, par.max_iters);
            par.softmax_temp = 0.08*(1.0 - t) + 0.03*t;

            employed_phase();
            (void)build_selection_probs(); // giữ ổn định RNG nếu cần
            onlooker_phase();

            bool stuck = (no_improve_iters >= par.stagnation_patience);
            scout_phase(stuck);

            if (par.cross_period > 0 && it % par.cross_period == 0)
                crossover_phase();

            update_best_and_pheromone();

            // // Debug (bật khi cần)
            // if (it % 50 == 0) {
            //     cerr << "[Iter " << it << "] best value=" << best.value
            //          << " weight=" << best.weight << " fitness=" << best.fitness
            //          << " limit=" << par.limit
            //          << " no_improve=" << no_improve_iters
            //          << " T=" << par.softmax_temp << "\n";
            // }
        }
    }
};





/*** ---------- Exact Solver: Branch & Bound with fractional UB ---------- ***/
// Upper bound (UB) bằng cách tham lam kiểu fractional knapsack trên phần còn lại
static inline double ub_frac_after(int i,
                                   const vector<long long>& w,
                                   const vector<long long>& v,
                                   long long curW, long long curV,
                                   long long Wmax) {
    if (curW > Wmax) return -1e100;
    double UB = (double)curV;
    long long Wleft = Wmax - curW;
    int n = (int)w.size();
    for (int k = i; k < n; ++k) {
        if (w[k] <= Wleft) {
            Wleft -= w[k]; UB += v[k];
        } else {
            UB += (double)v[k] * ((double)Wleft / (double)w[k]);
            break;
        }
    }
    return UB;
}

// Branch & Bound: trả về nghiệm TỐI ƯU (nếu duyệt xong)
static inline Solution branch_and_bound_knapsack(const Knapsack& P, const Solution& warmLB) {
    const int n = P.n;

    // Sắp xếp item theo mật độ v/w giảm dần (tăng độ sắc bén của UB)
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int a, int b){
        // tránh chia 0
        double da = (double)P.v[a] / (double)max(1LL, P.w[a]);
        double db = (double)P.v[b] / (double)max(1LL, P.w[b]);
        if (fabs(da - db) < 1e-12) return P.v[a] > P.v[b];
        return da > db;
    });
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; ++i) { w[i] = P.w[ord[i]]; v[i] = P.v[ord[i]]; }

    // Lower bound ban đầu từ ABC (nếu khả thi)
    long long bestV = 0;
    if (warmLB.weight <= P.W) bestV = max(bestV, warmLB.value);

    // Duyệt DFS với stack (ưu tiên nhánh "lấy" trước để nâng LB sớm)
    struct Frame {
        int i; long long Wsum, Vsum;
        // vector x theo thứ tự đã sắp (để reconstruct)
        vector<uint8_t> x;
    };
    vector<uint8_t> x0(n, 0);
    vector<Frame> st;
    st.push_back({0, 0, 0, x0});

    // Lưu nghiệm tốt nhất theo thứ tự sắp
    vector<uint8_t> best_x_sorted = x0;

    while (!st.empty()) {
        Frame cur = std::move(st.back()); st.pop_back();

        // Cắt tỉa bằng upper bound
        double UB = ub_frac_after(cur.i, w, v, cur.Wsum, cur.Vsum, P.W);
        if (UB <= (double)bestV + 1e-9) continue;

        if (cur.i == n) {
            if (cur.Wsum <= P.W && cur.Vsum > bestV) {
                bestV = cur.Vsum;
                best_x_sorted = cur.x;
            }
            continue;
        }

        // NHÁNH "LẤY" item i (nếu còn chỗ), push trước để ưu tiên
        if (cur.Wsum + w[cur.i] <= P.W) {
            Frame take = cur;
            take.x[cur.i] = 1;
            take.Wsum += w[cur.i];
            take.Vsum += v[cur.i];
            st.push_back({take.i + 1, take.Wsum, take.Vsum, std::move(take.x)});
        }

        // NHÁNH "KHÔNG LẤY" item i
        Frame skip = cur;
        skip.x[skip.i] = 0;
        st.push_back({skip.i + 1, skip.Wsum, skip.Vsum, std::move(skip.x)});
    }

    // Map nghiệm tốt nhất về chỉ số gốc
    Solution ans; ans.x.assign(n, 0);
    for (int i = 0; i < n; ++i) if (best_x_sorted[i]) ans.x[ ord[i] ] = 1;

    long long wsum = 0, vsum = 0;
    for (int i = 0; i < n; ++i) if (ans.x[i]) { wsum += P.w[i]; vsum += P.v[i]; }
    ans.weight = wsum; ans.value = vsum;
    ans.fitness = (wsum <= P.W ? (double)vsum : (double)vsum - 1e6*(wsum - P.W));
    return ans;
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long W;
    if (!(cin >> n >> W)) {
        // Demo ngẫu nhiên khi không có stdin
        n = 80; W = 600;
        Knapsack P(n, W);
        for (int i = 0; i < n; ++i) {
            long long wi = 1 + rng.randint(1, 20);
            long long vi = 1 + rng.randint(5, 100);
            P.w[i] = wi; P.v[i] = vi;
        }
        Params par; // dùng mặc định: colony_size=600, max_iters=250...
        ABC abc(P, par);
        abc.run();

        // BnB để nâng nghiệm (hoặc chứng minh tối ưu)
        Solution exact = branch_and_bound_knapsack(P, abc.best);
        const Solution& final_sol = (exact.value > abc.best.value && exact.weight <= P.W) ? exact : abc.best;

        // Output theo yêu cầu (dùng final_sol)
        int total_choose = 0;
        for (int i = 0; i < n; i++) if (final_sol.x[i]) total_choose++;
        cout << total_choose << '\n';
        for (int i = 0; i < n; i++) if (final_sol.x[i]) cout << i + 1 << " ";
        cout << '\n';
        return 0;
    } else {
        Knapsack P(n, W);
        for (int i = 0; i < n; ++i) {
            long long wi, vi; cin >> wi >> vi;
            P.w[i] = wi; P.v[i] = vi;
        }
        Params par; // giữ mặc định tối ưu đã chỉnh ở trên
        ABC abc(P, par);
        abc.run();

        // BnB để nâng nghiệm (hoặc chứng minh tối ưu)
        Solution exact = branch_and_bound_knapsack(P, abc.best);
        const Solution& final_sol = (exact.value > abc.best.value && exact.weight <= P.W) ? exact : abc.best;

        // Output theo yêu cầu (dùng final_sol)
        int total_choose = 0;
        for (int i = 0; i < n; i++) if (final_sol.x[i]) total_choose++;
        cout << total_choose << '\n';
        for (int i = 0; i < n; i++) if (final_sol.x[i]) cout << i + 1 << " ";
        cout << '\n';
    }
    return 0;
}
