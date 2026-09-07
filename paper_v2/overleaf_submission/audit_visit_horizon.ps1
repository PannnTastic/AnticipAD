param(
    [string]$Dataset = (Join-Path $PSScriptRoot '../../ADNI_POMDP_Complete.csv')
)

$ErrorActionPreference = 'Stop'
$rows = @(Import-Csv -LiteralPath $Dataset)
$clean = @($rows | Where-Object {
    $_.DX -notin @('', 'NA') -and $_.MMSE -notin @('', 'NA')
})

function Get-VisitSummary($Records) {
    $groups = @($Records | Group-Object RID)
    [ordered]@{
        records = $Records.Count
        participants = $groups.Count
        maximum_records_per_participant = [int](
            $groups | ForEach-Object { $_.Count } | Measure-Object -Maximum
        ).Maximum
        participants_with_20_records = @($groups | Where-Object Count -eq 20).Count
        maximum_distinct_dated_visits_per_participant = [int](
            $groups | ForEach-Object {
                @($_.Group.DATE | Where-Object { $_ -notin @('', 'NA') } |
                    Sort-Object -Unique).Count
            } | Measure-Object -Maximum
        ).Maximum
        missing_dates = @($Records | Where-Object DATE -in @('', 'NA')).Count
        repeated_rid_date_records = $Records.Count - @(
            $Records | Group-Object RID, DATE
        ).Count
    }
}

# Match the notebook's per-participant, date-sorted consecutive pairing.
# Undated records remain in the sequence, sorted last, but cannot give gaps.
$missingDatePairs = 0
$gaps = @($clean | Group-Object RID | ForEach-Object {
    $ordered = @($_.Group | Sort-Object DATE)
    for ($i = 1; $i -lt $ordered.Count; $i++) {
        $before = $ordered[$i - 1].DATE
        $after = $ordered[$i].DATE
        if ($before -in @('', 'NA') -or $after -in @('', 'NA')) {
            $missingDatePairs++
            continue
        }
        ([datetime]::ParseExact($after, 'yyyy-MM-dd',
            [cultureinfo]::InvariantCulture) -
         [datetime]::ParseExact($before, 'yyyy-MM-dd',
            [cultureinfo]::InvariantCulture)).Days
    }
} | Sort-Object)
$middle = [int][math]::Floor($gaps.Count / 2)
$median = if ($gaps.Count % 2) { $gaps[$middle] } else {
    ($gaps[$middle - 1] + $gaps[$middle]) / 2
}

[ordered]@{
    dataset = (Split-Path $Dataset -Leaf)
    sha256 = (Get-FileHash -LiteralPath $Dataset -Algorithm SHA256).Hash
    cleaning_rule = 'Exclude rows missing DX or MMSE, as in the preprocessing notebook'
    assembled = Get-VisitSummary $rows
    retained = Get-VisitSummary $clean
    retained_consecutive_pairs = [ordered]@{
        total = $gaps.Count + $missingDatePairs
        dated_pairs = $gaps.Count
        pairs_with_missing_date = $missingDatePairs
        minimum_days = $gaps[0]
        median_days = $median
        maximum_days = $gaps[-1]
        pairs_between_150_and_215_days_inclusive = @(
            $gaps | Where-Object { $_ -ge 150 -and $_ -le 215 }
        ).Count
    }
} | ConvertTo-Json -Depth 5
