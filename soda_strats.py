import itertools
import os.path
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

stocks = [14_184, 28_358, 42_552]

vip_minutes = [15, 3, 12, 4, 4, 1, 1, 1, 20]
vip_types = ["construction_worker", "real_estate", "delivery", "big_spender", "billionaire", "celeb", "influencer",
             "leprechaun"]


class StockedTower:
    def __init__(self, n_dreamers=0, n_login=1, avg_time=0, gt_level=1):
        nine_am = 9 * 60 * 60
        nine_pm = 21 * 60 * 60
        self.midnight = 24 * 60 * 60
        self.avg_time = avg_time

        self.n_login = n_login
        login_space = round((nine_pm - nine_am) / self.n_login)
        assert avg_time < login_space
        self.login_times = [t for t in range(nine_am, nine_pm, login_space)]
        self.logout_times = [t + avg_time for t in self.login_times]

        self.next_vip = nine_am + self.rand_vip_offset()

        self.max_stocks = [i for i in stocks]
        self.curr_stocks = [0 for _ in self.max_stocks]
        for i in range(n_dreamers):
            self.max_stocks[i] *= 2

        self.gt_level = gt_level
        self.sell_price = [1, 2, 3]
        if gt_level == 2:
            self.sell_price = [i * 1.1 for i in self.sell_price]
        elif gt_level == 3:
            self.sell_price = [i * 1.15 for i in self.sell_price]

        self.lobby = []
        self.day = 0
        self.curr_time = 0
        self.state = 'offline'
        self.total_earnings = 0

    def do_sim(self):
        while self.day < 7:
            self.do_day()
            self.day += 1

    def do_day(self):
        while self.curr_time < self.midnight:
            self.update()
            self.curr_time += 1

    def update(self):
        if self.state == 'offline':
            self.update_offline()
        elif self.state == 'online':
            self.update_online()

    def update_offline(self):
        if self.curr_time in self.login_times:
            self.state = 'online'
            return
        self.update_sell()

    def update_online(self):
        if self.curr_time in self.logout_times:
            self.state = 'offline'
            return
        self.update_restock()
        self.update_sell()
        self.update_get_vip()
        self.update_use_vip()

    def update_sell(self):
        if self.curr_time % 5 == 0 and self.curr_stocks[0] > 0:
            self.curr_stocks[0] -= 1
            self.total_earnings += self.sell_price[0]
        if self.curr_time % 10 == 0 and self.curr_stocks[1] > 0:
            self.curr_stocks[1] -= 1
            self.total_earnings += self.sell_price[1]
        if self.curr_time % 15 == 0 and self.curr_stocks[2] > 0:
            self.curr_stocks[2] -= 1
            self.total_earnings += self.sell_price[2]

    def new_vip(self):
        vip = random.choice(vip_types)
        if vip in ['construction_worker', 'real_estate', 'celeb', 'influencer', 'leprechaun']:
            return
        if vip == 'delivery' and vip not in self.lobby:
            self.lobby.append(vip)
        else:
            self.lobby.append(vip)

    @staticmethod
    def rand_vip_offset():
        return random.choice(vip_minutes) * 60

    def update_restock(self):
        for i, s in enumerate(self.curr_stocks):
            if s == 0:
                self.curr_stocks[i] = self.max_stocks[i]

    def update_get_vip(self):
        if self.next_vip == self.curr_time:
            self.new_vip()
            self.next_vip = self.curr_time + self.rand_vip_offset()
        elif self.next_vip < self.curr_time:
            self.next_vip = self.curr_time + self.rand_vip_offset()

    def update_use_vip(self):
        if not ('billionaire' in self.lobby or 'big_spender' in self.lobby):
            return
        if 'delivery' in self.lobby:
            self.use_delivery()
        vip = self.lobby.pop()
        if vip == 'billionaire':
            for price, stock in zip(self.sell_price, self.curr_stocks):
                self.total_earnings += price * stock
            self.curr_stocks = [0 for _ in self.curr_stocks]
        elif vip == 'big_spender':
            ind = random.choice([0, 1, 2])
            self.total_earnings += self.sell_price[ind] * self.curr_stocks[ind]
            self.curr_stocks[ind] = 0

    def use_delivery(self):
        self.lobby.remove('delivery')
        self.curr_stocks = [i for i in self.max_stocks]


class UnstockedTower(StockedTower):
    def update_offline(self):
        if self.curr_time in self.login_times:
            self.state = 'online'
            return
        # self.update_sell()

    def update_online(self):
        if self.curr_time in self.logout_times:
            self.state = 'offline'
            return
        self.update_get_vip()
        self.update_use_vip()

    def update_use_vip(self):
        if not self.lobby:
            return
        vip = self.lobby.pop()
        if vip == 'billionaire':
            for price, stock in zip(self.sell_price, self.max_stocks):
                self.total_earnings += price * stock
        elif vip == 'big_spender':
            self.total_earnings += self.sell_price[2] * self.max_stocks[2]


def plot_n_stocked(n):
    profits = []
    for i in tqdm(range(n)):
        t = StockedTower(n_dreamers=3, n_login=3, avg_time=15 * 60)
        t.do_sim()
        profits.append(t.total_earnings)
    plot_histogram(profits)


def plot_n_stocked_vs_unstocked(n):
    stocked = []
    unstocked = []
    for i in tqdm(range(n)):
        s = StockedTower(n_dreamers=3, n_login=3, avg_time=15 * 60, gt_level=2)
        s.do_sim()
        stocked.append(s.total_earnings)
        u = UnstockedTower(n_dreamers=3, n_login=3, avg_time=15 * 60, gt_level=2)
        u.do_sim()
        unstocked.append(u.total_earnings)
    dual_boxplots(stocked, unstocked)


def plot_histogram(ns):
    sns.histplot(ns, kde=True)
    plt.show()


def dual_histogram(xs, ys):
    sns.histplot(xs, bins=10, color='blue', kde=True, label='Series 1')
    sns.histplot(ys, bins=10, color='orange', kde=True, label='Series 2')
    plt.legend()
    plt.show()


def dual_boxplots(data1, data2):
    df = pd.DataFrame({
        'Stocked': data1,
        'Unstocked': data2
    })
    df_melted = df.melt(var_name='Series', value_name='Value')
    sns.boxplot(x='Series', y='Value', data=df_melted)
    plt.legend()
    plt.show()


def plot_n_logins(n):
    xs = [x for x in range(1, 10 + 1) for _ in range(n)]
    y1s = []
    y2s = []
    for login in tqdm(xs):
        s = StockedTower(n_dreamers=3, n_login=login, avg_time=15 * 60)
        s.do_sim()
        y1s.append(s.total_earnings)

        u = UnstockedTower(n_dreamers=3, n_login=login, avg_time=15 * 60)
        u.do_sim()
        y2s.append(u.total_earnings)
    df1 = pd.DataFrame({'x': xs, 'y': y1s, 'Series': 'Stocked'})
    df2 = pd.DataFrame({'x': xs, 'y': y2s, 'Series': 'Unstocked'})
    df = pd.concat([df1, df2])
    # Calculate means and standard deviations
    mean_df = df.groupby(['x', 'Series']).y.mean().reset_index()
    std_df = df.groupby(['x', 'Series']).y.std().reset_index()
    mean_df['yerr'] = std_df['y']

    # Create the scatterplot with error bars
    sns.lineplot(x='x', y='y', hue='Series', data=df)
    plt.show()


def plot_time(n):
    xs = [x for x in range(5, 60 + 1, 5) for _ in range(n)]
    y1s = []
    y2s = []
    for time in tqdm(xs):
        s = StockedTower(n_dreamers=3, n_login=3, avg_time=time * 60)
        s.do_sim()
        y1s.append(s.total_earnings)

        u = UnstockedTower(n_dreamers=3, n_login=3, avg_time=time * 60)
        u.do_sim()
        y2s.append(u.total_earnings)
    df1 = pd.DataFrame({'x': xs, 'y': y1s, 'Series': 'Stocked'})
    df2 = pd.DataFrame({'x': xs, 'y': y2s, 'Series': 'Unstocked'})
    df = pd.concat([df1, df2])
    # Calculate means and standard deviations
    mean_df = df.groupby(['x', 'Series']).y.mean().reset_index()
    std_df = df.groupby(['x', 'Series']).y.std().reset_index()
    mean_df['yerr'] = std_df['y']

    # Create the scatterplot with error bars
    sns.lineplot(x='x', y='y', hue='Series', data=df)
    plt.show()


def plot_dreamers(n):
    xs = [x for x in range(0, 3 + 1) for _ in range(n)]
    y1s = []
    y2s = []
    for n in tqdm(xs):
        s = StockedTower(n_dreamers=n, n_login=3, avg_time=15 * 60)
        s.do_sim()
        y1s.append(s.total_earnings)

        u = UnstockedTower(n_dreamers=n, n_login=3, avg_time=15 * 60)
        u.do_sim()
        y2s.append(u.total_earnings)
    df1 = pd.DataFrame({'x': xs, 'y': y1s, 'Series': 'Stocked'})
    df2 = pd.DataFrame({'x': xs, 'y': y2s, 'Series': 'Unstocked'})
    df = pd.concat([df1, df2])
    # Calculate means and standard deviations
    mean_df = df.groupby(['x', 'Series']).y.mean().reset_index()
    std_df = df.groupby(['x', 'Series']).y.std().reset_index()
    mean_df['yerr'] = std_df['y']

    # Create the scatterplot with error bars
    sns.lineplot(x='x', y='y', hue='Series', data=df)
    plt.show()


def plot_gt_levels(n):
    xs = [x for x in range(1, 3 + 1) for _ in range(n)]
    y1s = []
    y2s = []
    for n in tqdm(xs):
        s = StockedTower(n_dreamers=3, n_login=3, avg_time=15 * 60, gt_level=n)
        s.do_sim()
        y1s.append(s.total_earnings)

        u = UnstockedTower(n_dreamers=3, n_login=3, avg_time=15 * 60, gt_level=n)
        u.do_sim()
        y2s.append(u.total_earnings)
    df1 = pd.DataFrame({'x': xs, 'y': y1s, 'Series': 'Stocked'})
    df2 = pd.DataFrame({'x': xs, 'y': y2s, 'Series': 'Unstocked'})
    df = pd.concat([df1, df2])
    # Calculate means and standard deviations
    mean_df = df.groupby(['x', 'Series']).y.mean().reset_index()
    std_df = df.groupby(['x', 'Series']).y.std().reset_index()
    mean_df['yerr'] = std_df['y']

    # Create the scatterplot with error bars
    sns.lineplot(x='x', y='y', hue='Series', data=df)
    plt.show()


def plot_time_over_the_day(n):
    time = 60  # minutes
    xs = [x for x in range(1, 10 + 1) for _ in range(n)]
    y1s = []
    y2s = []
    for n_visits in tqdm(xs):
        s = StockedTower(n_dreamers=3, n_login=n_visits, avg_time=(time * 60) // n_visits)
        s.do_sim()
        y1s.append(s.total_earnings)

        u = UnstockedTower(n_dreamers=3, n_login=n_visits, avg_time=(time * 60) // n_visits)
        u.do_sim()
        y2s.append(u.total_earnings)
    df1 = pd.DataFrame({'x': xs, 'y': y1s, 'Series': 'Stocked'})
    df2 = pd.DataFrame({'x': xs, 'y': y2s, 'Series': 'Unstocked'})
    df = pd.concat([df1, df2])
    # Calculate means and standard deviations
    mean_df = df.groupby(['x', 'Series']).y.mean().reset_index()
    std_df = df.groupby(['x', 'Series']).y.std().reset_index()
    mean_df['yerr'] = std_df['y']

    # Create the scatterplot with error bars
    sns.lineplot(x='x', y='y', hue='Series', data=df)
    plt.show()


def get_all_data(n):
    filename = f'soda-data-{n}tests.csv'
    if not os.path.isfile(filename):
        calc_all_data(n, filename)
    return pd.read_csv(filename)


def calc_all_data(n, filename):
    # n_dreamers=0, n_login=1, avg_time=0, gt_level=1
    parameter_values = [
        list(range(n)),  # trial no
        list(range(3 + 1)),  # n_dreamers
        list(range(1, 10 + 1)),  # n_logins
        list(range(1 * 60, (60 * 60) + 1, 5 * 60)),  # avg_time (s)
        list(range(1, 3 + 1)),  # gt_level
    ]
    tests = list(itertools.product(*parameter_values))
    dream_vals, login_vals, time_vals, gt_vals, earnings_vals, strats = [], [], [], [], [], []
    for (_, n_dreamers, n_logins, avg_time, gt_level) in tqdm(tests):
        s = StockedTower(n_dreamers=n_dreamers, n_login=n_logins, avg_time=avg_time, gt_level=gt_level)
        s.do_sim()
        dream_vals.append(n_dreamers)
        login_vals.append(n_logins)
        time_vals.append(avg_time)
        gt_vals.append(gt_level)
        earnings_vals.append(s.total_earnings)
        strats.append('stocked')

        s = UnstockedTower(n_dreamers=n_dreamers, n_login=n_logins, avg_time=avg_time, gt_level=gt_level)
        s.do_sim()
        dream_vals.append(n_dreamers)
        login_vals.append(n_logins)
        time_vals.append(avg_time)
        gt_vals.append(gt_level)
        earnings_vals.append(s.total_earnings)
        strats.append('unstocked')
    total_time = [t * v for t, v in zip(time_vals, login_vals)]
    df = pd.DataFrame({'dream_vals': dream_vals,
                       'login_vals': login_vals,
                       'time_vals': time_vals,
                       'gt_vals': gt_vals,
                       'earnings_vals': earnings_vals,
                       'total_time': total_time,
                       'strats': strats})
    df.to_csv(filename, index=False)


def pairplot(n):
    df = get_all_data(n)

    # sns.violinplot(data=df, y='earnings_vals', x='dream_vals', hue='strats', split=True, inner='quart')
    # plt.show()
    #
    # sns.violinplot(data=df, y='earnings_vals', x='login_vals', hue='strats', split=True, inner='quart')
    # plt.show()

    sns.lineplot(data=df, y='earnings_vals', x='time_vals', hue='strats')
    plt.show()

    sns.lineplot(data=df, y='earnings_vals', x='total_time', hue='strats')
    plt.show()


def main():
    # plot_n_stocked_vs_unstocked(256)
    pairplot(5)


if __name__ == '__main__':
    main()
