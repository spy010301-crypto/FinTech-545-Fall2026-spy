"""Build concise written responses from the actual, unrounded calculation results."""

import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image

ROOT = Path(__file__).resolve().parent


def number(value, digits=4):
    return f'{value:.{digits}f}'


def percent(value, digits=3):
    return f'{100 * value:.{digits}f}%'


def pair(risk, scale=1, digits=2):
    return f"{scale * risk['VaR']:.{digits}f}/{scale * risk['ES']:.{digits}f}"


def sequence(values, digits=4, scale=1):
    return ', '.join(number(scale * value, digits) for value in values)


def make_pages(results):
    p1, p2, p3, p4, p5 = results
    pages = []
    comparisons = p1['parity_iv_comparison'][0]
    parity = '; '.join(f"{row['Strike']:.0f}: {row['gap']:+.4f}/{row['band']:.2f}" for row in p1['parity'])
    pages.append(('Problem 1 - Implied volatility from a dirty chain', [
        '<b>Predict (a-c).</b> I expected low-vega wings to have unstable IVs, and the three identical $0.05/$0.10 call quotes above 210 to create an artificial upturn. Before inversion, parity flags strike 175. A stale or mismatched quote is more plausible than a genuine volatility difference; parity alone does not tell us which leg is wrong.',
        f'Using T = 35/365 and carry r-q, gap = Cmid - Pmid - S exp(-qT) + K exp(-rT). Each entry below is strike: signed gap/full combined spread, in dollars. {parity}. Only |gap| &gt; band is flagged; this is the assignment\'s deliberately conservative spread test.',
        'Exact parity implies equal call/put IV: the GBSM call-minus-put price is independent of volatility, and each price increases with volatility. Passing a spread band is weaker than exact parity, so small IV differences at unflagged strikes are possible.',
        '<b>Fit (d-e).</b> Appendix A reports every bid/mid/ask IV, each mid-price vega, and the local IV change for a $0.01 price increase. Appendix C plots both smiles and bid/ask IVs. IV is obtained by root-finding on our own GBSM; no option-pricing package is used.',
        'Nine bids are below their discounted-intrinsic lower bound: calls at 140, 145, 150, 155 and puts at 205, 210, 215, 220, 225. No nonnegative-volatility price can match these bids. All mids and asks have a finite IV. Vega is dollars per 1.00 change in volatility; one-cent IV movement in percentage points is 100 x 0.01/vega.',
        f"<b>Reconcile (f).</b> Strike 175 appears as a put-IV dip: call IV 26.0457%, put IV 21.1620%. Using average call/put vega {comparisons['reference_vega']:.4f}, its ${comparisons['gap']:.4f} parity gap predicts {comparisons['approximate_iv_gap_points']:.4f} vol points; the actual gap is {comparisons['actual_iv_gap_points']:.4f}. The linear vega calculation explains the discrepancy closely, not exactly, because vega changes with IV.",
        '<b>Reconcile (g-h).</b> Above 210, constant call mids of $0.075 imply rising IVs of 23.8225%, 26.6888%, 29.4473%. A cent moves IV by about 0.443, 0.482, 0.519 points, and each bid/ask IV interval spans roughly 2.3-2.7 points. The puts have even lower vega and very wide IV ranges. This is weak price information, not convincing evidence of a market upturn. Parity misses it because dollar errors of $0.0355-$0.0698 lie inside $1.05-$1.36 combined spreads. Excluding strike 175 and the contaminated call wing, IV falls from about 39% at 140 toward 22% near 210: downside protection is priced at higher volatility than upside exposure. This is consistent with stronger downside demand, not proof of its cause.',
    ]))

    call, put = p2['prices']['call'], p2['prices']['put']
    converge = '; '.join(f"N={row['steps']}: {row['american_call']:.4f}/{row['american_put']:.4f}" for row in p2['convergence'])
    greek_lines = []
    for kind in ['call', 'put']:
        greek_lines.append(kind + ' ' + '; '.join(f"h={bump}: {g['delta']:.6f}/{g['gamma']:.6f}" for bump, g in p2['stock_bump_greeks'][kind].items()))
    dividend_lines = '; '.join(f"{name.replace('_', ' ')}: {sequence(values, 6)}" for name, values in p2['dividend_sensitivities'].items())
    pages.append(('Problem 2 - American options and discrete dividends', [
        f"<b>Predict (a-b).</b> The exercise hurdles are ${p2['dividend_hurdles'][0]:.6f} and ${p2['dividend_hurdles'][1]:.6f}. Dividend 1 ($0.25) fails; dividend 2 ($1.50) passes. I expected call exercise to be possible just before dividend 2 at sufficiently high stock prices. I predicted lower European/American call values, higher put values, a larger call exercise premium and a smaller put exercise premium: waiting for the stock drop benefits the put holder.",
        f"<b>Fit (c).</b> Without dividends, GBSM European call/put = ${call['european_no_dividend_gbsm']:.6f}/${put['european_no_dividend_gbsm']:.6f}; N=300 American tree call/put = ${call['american_no_dividend_tree']:.6f}/${put['american_no_dividend_tree']:.6f}. With dividends, European tree call/put = ${call['european_dividend_tree']:.6f}/${put['european_dividend_tree']:.6f}; American tree call/put = ${call['american_dividend_tree']:.6f}/${put['american_dividend_tree']:.6f}.",
        f"Subtracting PV(dividends) = ${p2['present_value_dividends']:.6f} from spot gives approximate GBSM European call/put ${call['european_pv_adjusted_gbsm']:.6f}/${put['european_pv_adjusted_gbsm']:.6f}, each ${-call['pv_adjusted_gbsm_minus_dividend_tree']:.6f} below the exact-dividend tree. PV subtraction changes the initial diffusion base; fixed cash jumps cannot generally be replaced by a smaller lognormal stock.",
        f'<b>Fit (d).</b> American call/put convergence: {converge}. The oscillation, including 100 versus 105, is normal lattice error. N=300 is the reported estimate, not an exact continuous-time price.',
        '<b>Fit (e).</b> Central delta/gamma at fixed N=100, with stock bumps in dollars: ' + '. '.join(greek_lines) + '. Tiny values printed as zero are numerical near-zero values.',
        '<b>Fit (f).</b> N=300 sensitivities per dollar of dividend 1/dividend 2, using central $0.01 bumps: ' + dividend_lines + '.',
        f"<b>Reconcile (g).</b> Matched-tree American-minus-European premiums without/with dividends: call ${p2['early_exercise_premiums']['call']['without_dividends_matched_tree']:.6f}/${p2['early_exercise_premiums']['call']['with_dividends_matched_tree']:.6f}; put ${p2['early_exercise_premiums']['put']['without_dividends_matched_tree']:.6f}/${p2['early_exercise_premiums']['put']['with_dividends_matched_tree']:.6f}. Both predictions held. Without dividends, American-minus-GBSM is ${p2['early_exercise_premiums']['call']['without_dividends_tree_minus_gbsm']:.6f} for the call and ${p2['early_exercise_premiums']['put']['without_dividends_tree_minus_gbsm']:.6f} for the put; the call difference is discretization, not exercise value. Just before dividend 2, exercise becomes optimal above about ${p2['call_exercise_boundary']['tree_N300']:.4f} (N=100: ${p2['call_exercise_boundary']['tree_N100']:.4f}; continuous post-dividend GBSM check: ${p2['call_exercise_boundary']['continuous_gbsm_after_dividend']:.4f}).",
        '<b>Reconcile (h-i).</b> Gamma is unreliable as the bump shrinks: a fixed-grid tree is piecewise linear in spot, with kinks where exercise or terminal payoffs switch. A tiny bump can miss every kink and return zero, or cross one and give an unstable number. This is not just floating-point error. The American call can exercise before dividend 2, largely avoiding that cash drop; its sensitivity to dividend 2 is therefore much smaller than to dividend 1, before which exercising is not worthwhile. The European call has no such escape.',
    ]))

    normal, student = p3['fitted_distributions']['normal'], p3['fitted_distributions']['t']
    books = p3['books']
    greek_text = '; '.join(f"{name}: {book['greeks']['delta']:.4f}, {book['greeks']['gamma']:.4f}, {book['greeks']['theta']:.4f}" for name, book in books.items())
    no_move = '; '.join(f"{name} ${book['no_move_pnl']:.2f}" for name, book in books.items())
    p3_paras = [
        '<b>Predict (a-b).</b> At S=$99.12, book delta/gamma/theta (theta per year) are: ' + greek_text + '. I ranked dollar delta-normal error: ShortStraddle, HedgedShortCall, CoveredCall, RiskReversal. Negative gamma suggests understatement for the first three; small positive local RiskReversal gamma suggests slight overstatement. Positive theta predicts positive no-move P&L in all four. The two nearly delta-neutral short-gamma books should have especially high ES/VaR ratios.',
        f"<b>Fit (c).</b> The 500 daily arithmetic returns have mean {p3['return_moments']['mean']:.8f}, sample variance {p3['return_moments']['variance']:.8f}, skew {p3['return_moments']['skewness']:.5f}, excess kurtosis {p3['return_moments']['excess_kurtosis']:.5f}. Normal MLE: mean {normal['mean']:.8f}, SD {normal['sd']:.8f}, AICc {normal['aicc']:.3f}. Student t MLE: df {student['df']:.5f}, location {student['location']:.8f}, scale {student['scale']:.8f} (SD {student['sd']:.8f}), AICc {student['aicc']:.3f}. The t wins; AICc counts 2/3 parameters.",
        '<b>Fit (d-f).</b> Dollar VaR/ES pairs below are ordered 5%, then 1%. DN uses sample daily SD, zero mean and sqrt(10). Normal (N) and t use their fitted scale/shape with daily mean set to zero, 1,000,000 paths, seed 1234, ten compounded daily returns, fixed IV and 20/255 years remaining. Appendix D plots full P&L against terminal stock with the delta-only line.',
    ]
    for name, book in books.items():
        n, t = book['normal']['full_revaluation'], book['t']['full_revaluation']
        p3_paras.append(f"<b>{name}.</b> Current value ${book['current_value']:.2f}. DN {pair(book['delta_normal']['5%'])}; {pair(book['delta_normal']['1%'])}. N mean ${n['mean']:.2f}, risk {pair(n['5%'])}; {pair(n['1%'])}. t mean ${t['mean']:.2f}, risk {pair(t['5%'])}; {pair(t['1%'])}.")
    p3_paras.extend([
        '<b>Reconcile (g).</b> Small delta does not remove gamma or theta: dP approximately equals delta*dS + 0.5*gamma*dS^2 + theta*dt. Under t, ShortStraddle/HedgedShortCall 5% delta-gamma (no theta) VaRs are $429.74/$214.87; adding theta gives $299.80/$141.61, versus full $321.65/$152.52. At 1%, delta-gamma-theta VaRs are $663.60/$323.51 and ES $967.61/$475.52, versus full $633.36/$308.38 and $844.46/$413.93. Curvature helps, but frozen Greeks cannot reproduce changing gamma and expiry. Exact no-move P&L: ' + no_move + '.',
        '<b>Reconcile (h-i).</b> Means are positive as predicted: short-gamma carry compensates large-move losses, not free profit. The error ranking holds for 5% VaR; RiskReversal can understate deep t-tail risk because its short put dominates after a large fall. Normal-to-t changes 5% VaR little, but raises 1% ES by about $99 for the straddle and $49 for the hedge. Similar fitted SDs and ten-day averaging keep moderate quantiles close; ES still sees rare jumps. Their t 5% ES/VaR ratios are 1.61/1.64; the zero-mean normal reference is 1.25 at 5%, not at every tail level.',
        '<b>Reply (j).</b> The book is nearly delta-flat, not risk-flat: its actual delta is -0.4444 and its negative gamma leaves losses after a large move in either direction. At ten days, the t-model 1% VaR/ES are about $308/$414, so limits should include full-revaluation tail risk rather than delta alone.',
    ])
    pages.append(('Problem 3 - VaR of an option book', p3_paras))

    p4_paras = [
        '<b>Predict (a-b).</b> Each asset below lists monthly mean, variance in decimal-return squared, skew, excess kurtosis, annual Sharpe, and mean SE in percentage points. These are sample variance and central standardized moments; RF=3%/12 per month. I expected over 50% PUTW in the long-only optimum and historical 1% ES well above a normal approximation.',
    ]
    for name, asset in p4['assets'].items():
        p4_paras.append(f"{name}: {percent(asset['mean'])}, {asset['variance']:.7f}, {asset['skewness']:.4f}, {asset['excess_kurtosis']:.4f}, {asset['sharpe_annual']:.4f}, SE {100*asset['mean_standard_error']:.4f} points.")
    putw = p4['assets']['PUTW']
    p4_paras.append(f"PUTW's {putw['worst_month']} loss of {percent(-putw['worst_return'])} supplies {percent(putw['worst_centered_variance_share'],2)} of centered squared deviations and {percent(putw['worst_centered_fourth_moment_share'],2)} of fourth powers. Removing it only as an influence check changes excess kurtosis to {putw['without_worst_moments']['excess_kurtosis']:.2f} and Sharpe to {putw['without_worst_sharpe']:.3f}. All months remain in the actual optimization.")
    p4_paras.append('<b>Fit (c-d).</b> Weights are percentages in EQ/BOND/STK/PUTW order. Risk numbers are monthly percentage-loss VaR/ES pairs at 5%, then 1%; H means historical and N same-mean/sample-SD normal. The frontier, CML and four assets appear in Appendix C.')
    for name, portfolio in p4['portfolios'].items():
        p4_paras.append(f"{name.replace('_',' ')}: weights {sequence(portfolio['weights'],3,100)}; annual Sharpe {portfolio['sharpe_annual']:.4f}. H {pair(portfolio['0.05']['historical'],100,3)}; {pair(portfolio['0.01']['historical'],100,3)}. N {pair(portfolio['0.05']['normal'],100,3)}; {pair(portfolio['0.01']['normal'],100,3)}.")
    capm, down = p4['capm'], p4['downside_capm']
    stability = p4['bootstrap']['weights']['long_only']
    p4_paras.extend([
        f"<b>Fit (e).</b> PUTW excess-return CAPM: monthly alpha {percent(capm['coefficients']['alpha'],4)}, alpha t {capm['t_statistics']['alpha']:.4f}, beta {capm['coefficients']['EQ']:.4f}, R-squared {capm['r_squared']:.4f}. Regressing separately in {down['n']} months with EQ excess return below -5% gives beta {down['coefficients']['EQ']:.4f}.",
        '<b>Reconcile (f-g).</b> Both predictions held. The optimizer bought premium income from selling crash insurance. Sharpe sees only mean and SD; it does not distinguish occasional catastrophic losses from symmetric noise. ES averages tail severity. Population/exact empirical ES is monotone, cash-translation invariant, positively homogeneous and subadditive, so diversification does not increase aggregate risk beyond the sum of stand-alone ES. Positive linear CAPM alpha is not proof of skill: full-sample beta about 0.18 hides downside beta about 0.94 and a nonlinear put payoff.',
        f"<b>Reconcile (h).</b> The mean SEs above are SD/sqrt(240), assuming independent months. They are large relative to the excess means, so precise tangency weights are not trustworthy. A six-month-block bootstrap (500 samples) gives long-only PUTW weights from {percent(stability['percentile_2_5'][3],2)} to {percent(stability['percentile_97_5'][3],2)} at the 2.5/97.5 percentiles. The 20% cap controls concentration; its stability is imposed, not statistical precision. Two unrestricted bootstrap samples have no finite positive tangency and are excluded, with counts retained in the results.",
        f"At 1%, the course historical ES uses the two worst months. Fractional empirical ES uses 2.4 months and gives long-only {percent(p4['portfolios']['long_only']['0.01']['fractional_tail_ES_sensitivity'])}, still far above normal {percent(p4['portfolios']['long_only']['0.01']['normal']['ES'])}. This distinction matters with only 240 months; it does not change the insurance-risk conclusion. Dates through 2026-11-30 are supplied exercise data, not a claim of already observed live performance.",
    ])
    pages.append(('Problem 4 - A Sharpe ratio that sells insurance', p4_paras))

    p5_paras = [
        '<b>Predict (a-b).</b> The funds cover 1996-10-01 to 2001-09-28, with 1,257 matched dates. Factors and RF are converted from percent to decimal. Initial CAPM made FUND1 look more skilled, but its large residual correlation with momentum suggested that adding momentum would remove much of its alpha. FUND2 should change less.',
    ]
    for name, fund in p5['funds'].items():
        capm = fund['models']['CAPM']
        p5_paras.append(f"{name}: daily mean excess {percent(fund['daily_mean_excess'],5)}, total-return SD {percent(fund['daily_total_volatility'],5)}, annual Sharpe {fund['sharpe_annual']:.4f}. CAPM daily alpha {capm['coefficients']['alpha']:.8f}, t {capm['t_statistics']['alpha']:.4f}. Residual correlations with SMB/HML/Mom: {sequence(fund['capm_residual_correlations'].values(),5)}. Sharpe uses excess-return SD; annualization uses 255 trading days.")
    p5_paras.append('<b>Fit (c).</b> Appendix B lists every coefficient and SE, alpha t, R-squared and AICc for all six regressions. SEs are conventional homoskedastic OLS. Gaussian AICc includes residual variance: 3 parameters for CAPM, 5 for FF3 and 6 for Carhart. Alpha is a daily decimal return; betas are unitless.')
    recent = p5['recent_factor_means_daily']
    p5_paras.append(f"<b>Fit (d).</b> The most recent decade in the factor files is (2014-09-30, 2024-09-30], containing 2,516 dates. Daily mean Mkt-RF/SMB/HML/Mom/RF = {sequence([recent[key] for key in ['Mkt-RF','SMB','HML','Mom','RF']],8)}. Annual total expectation is 255 x (mean RF + betas x mean factors), adding 255 x fitted alpha only in the with-alpha case.")
    for name, fund in p5['funds'].items():
        p5_paras.append(f"{name}: expected annual without/with alpha {percent(fund['expected_annual_without_alpha'])}/{percent(fund['expected_annual_with_alpha'])}; realized annual arithmetic return {percent(fund['realized_annual_arithmetic'])}. These later factor premia are a hypothetical repricing of older exposures, not an out-of-sample historical forecast.")
    stress = p5['worst_momentum_entire_file']
    p5_paras.append(f"<b>Fit (e).</b> The entire momentum file's worst day is {stress['date']}, return {percent(stress['return'],2)}. With other factors flat, the isolated shock is momentum beta times that return; alpha, RF and residual shocks are not added.")
    for name, fund in p5['funds'].items():
        shock = fund['momentum_stress']
        p5_paras.append(f"{name}: momentum beta {fund['models']['Carhart4']['coefficients']['Mom']:.6f}; $10 million loss ${shock['loss_on_10million']:,.2f}, equal to {shock['loss_daily_sd_multiple']:.4f} sample daily standard deviations.")
    fund1, fund2 = p5['funds']['FUND1'], p5['funds']['FUND2']
    sample = p5['sample_factor_means_daily']
    bridge = fund1['alpha_difference_bridge_daily']
    p5_paras.extend([
        '<b>Reconcile (f).</b> The predictions held. FUND1 CAPM/FF3/Carhart daily alphas are ' + sequence([fund1['models'][key]['coefficients']['alpha'] for key in ['CAPM','FF3','Carhart4']],8) + '; FUND2 alphas are ' + sequence([fund2['models'][key]['coefficients']['alpha'] for key in ['CAPM','FF3','Carhart4']],8) + '.',
        f"Sample daily market/SMB/HML/Mom premia are {sequence([sample[key] for key in ['Mkt-RF','SMB','HML','Mom']],8)}. FUND1's CAPM-minus-Carhart alpha = {10000*fund1['capm_alpha_minus_carhart_alpha']:.6f} daily basis points, decomposed into market-beta change/SMB/HML/Mom = {sequence(bridge.values(),6,10000)}. The omitted momentum and value exposures explain almost all the apparent alpha.",
        '<b>Reconcile (g-h).</b> Under APT, alpha is unexplained expected excess return relative to the chosen factors, not an absolute certificate of skill. Opinion: FUND2 is the more defensible alpha-fee candidate, but its four-factor t around 2.15 is modest evidence, not proof of persistence. I would use Carhart to screen omitted momentum; FF3 has slightly better AICc for FUND2 because momentum adds little. FUND1\'s high sample return relied on factor premia that do not carry over to the later decade. Factor models clarify exposures, but optimizer inputs still depend on noisy premia and stable betas/alpha; do not mechanically extrapolate historical returns.',
    ])
    pages.append(('Problem 5 - Alpha or factor exposure?', p5_paras))

    appendix = [
        'For each strike: call and put bid/mid/ask IVs (%); vega (dollars per 1.00 volatility); one-cent IV movement (volatility percentage points). NA means the bid is below the no-arbitrage lower bound. All displayed results are rounded; CSV and JSON retain full precision.',
    ]
    for strike in sorted({quote['strike'] for quote in p1['quotes']}):
        legs = []
        for kind in ['Call','Put']:
            quote = next(q for q in p1['quotes'] if q['strike'] == strike and q['type'] == kind)
            vols = '/'.join('NA' if quote[key] is None else f'{100*quote[key]:.4f}' for key in ['bid_iv','mid_iv','ask_iv'])
            legs.append(f"{kind} IV {vols}; vega {quote['vega']:.4f}; cent {quote['cent_iv_points']:.4f}")
        appendix.append(f"<b>K={strike:.0f}.</b> " + ' | '.join(legs) + '.')
    appendix.extend([
        'Numerical conventions: sample variance uses n-1; skewness/excess kurtosis are standardized central moments. Historical VaR uses the average floor/ceiling order statistics at alpha*n (1-based); ES is minus the mean below that cutoff. Negative values would mean gains, not values to convert with abs().',
        'Problem 3 simulation uncertainty: results are estimates. With ten independent simulation batches, t-model 1% ES Monte Carlo SE is about $3.83 for ShortStraddle and $1.92 for HedgedShortCall. Arithmetic normal/t models have unbounded downside; no <=-100% return occurred in these seeded draws, and none was clipped or retried. This tail-support limitation remains a model assumption.',
    ])
    pages.append(('Appendix A - Every option IV and sensitivity', appendix))

    appendix = ['Each coefficient is estimate (standard error). Alpha is daily decimal return; all other coefficients are unitless. The intercept is included in every regression.']
    for name, fund in p5['funds'].items():
        for model_name, model in fund['models'].items():
            coefficients = '; '.join(f"{key} {value:.8f} ({model['standard_errors'][key]:.8f})" for key,value in model['coefficients'].items())
            appendix.append(f"<b>{name}, {model_name}.</b> {coefficients}. Alpha t {model['t_statistics']['alpha']:.5f}; R-squared {model['r_squared']:.6f}; AICc {model['aicc']:.5f}.")
    appendix.append('Conventional SEs and the reported t statistics are conditional on OLS error assumptions; they do not establish causal skill or guarantee future alpha. The isolated momentum scenario is a factor stress, not the complete historical 1939 market scenario.')
    pages.append(('Appendix B - Complete regression estimates', appendix))
    return pages


def main():
    results = [json.loads((ROOT / 'results' / f'problem{i}_results.json').read_text()) for i in range(1,6)]
    pages = make_pages(results)
    body = ParagraphStyle('Body', fontName='Helvetica', fontSize=9.5, leading=12.2, spaceAfter=6)
    appendix_body = ParagraphStyle('Appendix', parent=body, fontSize=9, leading=11.2, spaceAfter=4)
    heading = ParagraphStyle('Heading', fontName='Helvetica-Bold', fontSize=15, leading=18, spaceAfter=12,
                             textColor=colors.HexColor('#183b56'))
    story, markdown = [], ['# Assignment 3 Responses', '']
    for index, (title, paragraphs) in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(title.replace('&', '&amp;'), heading))
        markdown.extend(['## ' + title, ''])
        for paragraph in paragraphs:
            escaped = re.sub(r'&(?![A-Za-z]+;|#\d+;)', '&amp;', paragraph).replace('<=', '&lt;=')
            story.append(Paragraph(escaped, appendix_body if index >= 5 else body))
            markdown.extend([re.sub(r'</?b>', '**', paragraph).replace('&gt;', '>'), ''])

    def add_picture(filename, width=510, max_height=300):
        path = ROOT / 'figures' / filename
        image_width, image_height = ImageReader(str(path)).getSize()
        ratio = min(width / image_width, max_height / image_height)
        story.append(Image(str(path), width=image_width*ratio, height=image_height*ratio))
        story.append(Spacer(1, 8))
        markdown.extend([f'![{filename}](figures/{filename})', ''])

    story.append(PageBreak())
    story.append(Paragraph('Appendix C - IV smile and efficient frontier', heading))
    markdown.extend(['## Appendix C - IV smile and efficient frontier', ''])
    add_picture('problem1_implied_volatility.png', max_height=260)
    add_picture('problem4_frontier.png', max_height=320)
    story.append(PageBreak())
    story.append(Paragraph('Appendix D - Option P&amp;L and insurance exposure', heading))
    markdown.extend(['## Appendix D - Option P&L and insurance exposure', ''])
    add_picture('problem3_pnl.png', max_height=350)
    add_picture('problem4_capm.png', max_height=255)

    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.gray)
        canvas.drawString(42, 24, 'FinTech 545 | Assignment 3 | All calculations reproducible from supplied data')
        canvas.drawRightString(570, 24, str(document.page))

    output = ROOT / 'Assignment3_Responses.pdf'
    document = SimpleDocTemplate(str(output), pagesize=(612,792), leftMargin=42, rightMargin=42,
                                 topMargin=38, bottomMargin=42,
                                 title='Assignment 3 Responses - Options, Risk and Portfolios',
                                 author='Shanpeiyuan')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / 'Assignment3_Responses.md').write_text('\n'.join(markdown))
    print(f'Built {output.name} and Assignment3_Responses.md')


if __name__ == '__main__':
    main()
