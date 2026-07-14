param(
    [string]$OutputPath = (Join-Path $PSScriptRoot '中期报告_I2V_CompBench.pptx')
)

$ErrorActionPreference = 'Stop'

function Color([string]$Hex) {
    $h = $Hex.TrimStart('#')
    $r = [Convert]::ToInt32($h.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($h.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($h.Substring(4, 2), 16)
    return $r + 256 * $g + 65536 * $b
}

$C = @{
    Navy      = Color '#14213D'
    Navy2     = Color '#1F3A5F'
    Blue      = Color '#2F6BFF'
    Cyan      = Color '#00A7B5'
    Orange    = Color '#F28C28'
    Red       = Color '#D64545'
    Green     = Color '#2E8B57'
    Gold      = Color '#F2C14E'
    Ink       = Color '#1F2937'
    Gray      = Color '#64748B'
    LightGray = Color '#E5EAF0'
    Panel     = Color '#F5F7FA'
    White     = Color '#FFFFFF'
    PaleBlue  = Color '#EAF1FF'
    PaleCyan  = Color '#E7F8FA'
    PaleOrange= Color '#FFF2E5'
    PaleRed   = Color '#FDECEC'
    PaleGreen = Color '#EAF7F0'
}

$msoTrue = -1
$msoFalse = 0
$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24

function Add-Text {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [string]$Text, [double]$Size = 18, [int]$ColorValue = $C.Ink,
          [bool]$Bold = $false, [int]$Align = 1, [string]$Font = 'Aptos',
          [double]$Margin = 2)
    $s = $Slide.Shapes.AddTextbox(1, [single]$X, [single]$Y, [single]$W, [single]$H)
    $s.TextFrame2.TextRange.Text = $Text
    $s.TextFrame2.WordWrap = $msoTrue
    $s.TextFrame2.AutoSize = 0
    $s.TextFrame2.MarginLeft = $Margin
    $s.TextFrame2.MarginRight = $Margin
    $s.TextFrame2.MarginTop = $Margin
    $s.TextFrame2.MarginBottom = $Margin
    $s.TextFrame2.TextRange.Font.Name = $Font
    $s.TextFrame2.TextRange.Font.Size = $Size
    $s.TextFrame2.TextRange.Font.Bold = $(if ($Bold) { $msoTrue } else { $msoFalse })
    $s.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $ColorValue
    $s.TextFrame2.TextRange.ParagraphFormat.Alignment = $Align
    return $s
}

function Add-Box {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [int]$Fill = $C.White, [int]$Line = $C.LightGray,
          [double]$Radius = 5, [double]$LineWeight = 1)
    $shapeType = $(if ($Radius -gt 0) { 5 } else { 1 })
    $s = $Slide.Shapes.AddShape($shapeType, [single]$X, [single]$Y, [single]$W, [single]$H)
    $s.Fill.ForeColor.RGB = $Fill
    $s.Fill.Transparency = 0
    $s.Line.ForeColor.RGB = $Line
    $s.Line.Weight = $LineWeight
    return $s
}

function Add-Line {
    param($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2,
          [int]$ColorValue = $C.Gray, [double]$Weight = 2, [bool]$Arrow = $false,
          [bool]$Dashed = $false)
    $s = $Slide.Shapes.AddLine([single]$X1, [single]$Y1, [single]$X2, [single]$Y2)
    $s.Line.ForeColor.RGB = $ColorValue
    $s.Line.Weight = $Weight
    if ($Arrow) { $s.Line.EndArrowheadStyle = 3 }
    if ($Dashed) { $s.Line.DashStyle = 4 }
    return $s
}

function Add-Placeholder {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [string]$Label, [string]$Hint = 'Delete this box and insert your image')
    $p = Add-Box $Slide $X $Y $W $H $C.White $C.Gray 0 1.5
    $p.Line.DashStyle = 4
    $p.Fill.Transparency = 1.0
    Add-Text $Slide ($X + 10) ($Y + 10) ($W - 20) 22 'IMAGE PLACEHOLDER' 11 $C.Gray $true 2 | Out-Null
    Add-Text $Slide ($X + 14) ($Y + $H/2 - 18) ($W - 28) 36 $Label 15 $C.Gray $true 2 | Out-Null
    Add-Text $Slide ($X + 12) ($Y + $H - 30) ($W - 24) 18 $Hint 9 $C.Gray $false 2 | Out-Null
    return $p
}

function Add-Card {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [string]$Title, [string]$Body, [int]$Accent = $C.Blue,
          [int]$Fill = $C.White, [double]$TitleSize = 16, [double]$BodySize = 12)
    Add-Box $Slide $X $Y $W $H $Fill $C.LightGray 5 1 | Out-Null
    $bar = Add-Box $Slide $X $Y 6 $H $Accent $Accent 0 0
    $bar.Line.Visible = $msoFalse
    Add-Text $Slide ($X + 18) ($Y + 12) ($W - 30) 26 $Title $TitleSize $C.Ink $true | Out-Null
    Add-Text $Slide ($X + 18) ($Y + 44) ($W - 30) ($H - 54) $Body $BodySize $C.Gray $false | Out-Null
}

function Add-Header {
    param($Slide, [string]$Title, [int]$Number, [string]$Kicker = 'MIDTERM PROGRESS REPORT')
    $Slide.FollowMasterBackground = $msoFalse
    $Slide.Background.Fill.ForeColor.RGB = $C.White
    $band = Add-Box $Slide 0 0 960 60 $C.Navy $C.Navy 0 0
    $band.Line.Visible = $msoFalse
    Add-Text $Slide 32 10 760 38 $Title 25 $C.White $true | Out-Null
    Add-Text $Slide 795 13 105 18 $Kicker 8 $C.LightGray $true 2 | Out-Null
    Add-Text $Slide 905 14 28 24 ([string]$Number) 13 $C.Gold $true 2 | Out-Null
    Add-Line $Slide 30 512 930 512 $C.LightGray 0.7 | Out-Null
    Add-Text $Slide 32 516 420 14 'I2V-CompBench · Candidate benchmark under validation' 8 $C.Gray | Out-Null
}

function New-ContentSlide {
    param($Presentation, [string]$Title, [int]$Number)
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, $ppLayoutBlank)
    Add-Header $slide $Title $Number
    return $slide
}

$app = $null
$presentation = $null
try {
    $app = New-Object -ComObject PowerPoint.Application
    $app.Visible = $msoTrue
    $app.DisplayAlerts = 1
    $presentation = $app.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    Write-Output 'Building slide 1...'
    # Slide 1
    $s = $presentation.Slides.Add(1, $ppLayoutBlank)
    $s.FollowMasterBackground = $msoFalse
    $s.Background.Fill.ForeColor.RGB = $C.Navy
    Add-Text $s 48 54 510 120 'Benchmark Dataset Construction for Compositional Image-to-Video Generation' 30 $C.White $true | Out-Null
    Add-Text $s 50 182 430 30 'Midterm Progress Report' 18 $C.Gold $true | Out-Null
    Add-Text $s 50 233 360 70 "[Name]`n[University / Lab] · 2026" 14 $C.LightGray | Out-Null
    Add-Placeholder $s 590 58 320 335 'Insert I2V cover example' 'Suggested: input image → generated frame sequence' | Out-Null
    Add-Line $s 50 420 910 420 $C.Navy2 1 | Out-Null
    Add-Text $s 50 442 860 36 'I2V  ·  COMPOSITIONAL EVALUATION  ·  DATA QUALITY  ·  VALIDATION' 11 $C.Cyan $true 2 | Out-Null

    Write-Output 'Building slide 2...'
    # Slide 2
    $s = New-ContentSlide $presentation 'Why Narrow Long-Video Benchmark to I2V?' 2
    Add-Text $s 34 73 892 24 'A common extension route repeatedly solves a short-horizon conditional generation problem.' 15 $C.Navy2 $true | Out-Null
    $clipX = @(48, 266, 484, 702)
    for ($i=0; $i -lt 4; $i++) {
        Add-Box $s $clipX[$i] 135 160 95 $C.PaleBlue $C.Blue 5 1.3 | Out-Null
        Add-Text $s ($clipX[$i]+12) 153 136 26 ("Short clip " + ($i+1)) 16 $C.Navy $true 2 | Out-Null
        Add-Text $s ($clipX[$i]+12) 187 136 18 'short-horizon rollout' 10 $C.Gray $false 2 | Out-Null
        if ($i -lt 3) {
            Add-Line $s ($clipX[$i]+162) 182 ($clipX[$i+1]-8) 182 $C.Orange 3 $true | Out-Null
            Add-Text $s ($clipX[$i]+154) 115 68 28 "tail frame(s)`n+ prompt" 9 $C.Orange $true 2 | Out-Null
        }
    }
    Add-Text $s 48 246 814 20 'StreamingT2V · ViD-GPT · Ca2-VDM · STIV' 11 $C.Blue $true 2 | Out-Null
    Add-Card $s 48 284 268 142 'Why I2V matters' "Each rollout includes a local visual-state transition. Identity drift, wrong action binding, and camera-motion confusion can propagate." $C.Cyan $C.PaleCyan 15 12
    Add-Card $s 346 284 268 142 'Method boundary' "Not all long-video methods repeat I2V. Queue, window-based, and hierarchical routes must be evaluated separately." $C.Orange $C.PaleOrange 15 12
    Add-Card $s 644 284 268 142 'Research decision' "Mechanism decomposition—not task equivalence. Validate the reusable local capability before system-level evaluation." $C.Green $C.PaleGreen 15 12
    Add-Text $s 48 453 864 34 'Literature: StreamingT2V (CVPR 2025) · ViD-GPT · Ca2-VDM (ICML 2025) · STIV (ICCV 2025)' 9 $C.Gray $false 2 | Out-Null

    Write-Output 'Building slide 3...'
    # Slide 3
    $s = New-ContentSlide $presentation 'I2V Motivation & Benchmark Gap' 3
    Add-Placeholder $s 38 86 518 350 'Insert the two-dog conceptual example' 'Keep the area for: input + wrong subject + zoom-only + correct output' | Out-Null
    Add-Text $s 38 445 518 28 'Conceptual illustration — not an experimental result' 10 $C.Gray $false 2 | Out-Null
    Add-Card $s 584 88 338 88 'Core problem' 'Visual quality does not guarantee instruction following.' $C.Red $C.PaleRed 15 13
    Add-Card $s 584 190 338 126 'Existing benchmark gap' "T2I: no temporal dynamics`nT2V: no first-frame constraint`nVBench: broad quality coverage" $C.Blue $C.PaleBlue 15 12
    Add-Card $s 584 330 338 116 'Our focus' "Explicit Preserve + Transform constraints for subject-level failure attribution." $C.Cyan $C.PaleCyan 15 13
    Add-Text $s 584 462 338 28 'Known first frame ⇒ change and preservation are jointly testable' 10 $C.Navy2 $true 2 | Out-Null

    Write-Output 'Building slide 4...'
    # Slide 4
    $s = New-ContentSlide $presentation 'Preserve–Transform Evaluation Framework' 4
    Add-Placeholder $s 42 106 170 170 'Insert one input image I₀' 'Optional example image' | Out-Null
    Add-Line $s 214 191 300 191 $C.Gray 2.5 $true | Out-Null
    Add-Box $s 302 100 250 90 $C.PaleCyan $C.Cyan 5 1.5 | Out-Null
    Add-Text $s 320 118 214 24 'PRESERVE SET  P' 16 $C.Cyan $true 2 | Out-Null
    Add-Text $s 320 151 214 24 'Identity · layout · background' 11 $C.Gray $false 2 | Out-Null
    Add-Box $s 302 212 250 90 $C.PaleOrange $C.Orange 5 1.5 | Out-Null
    Add-Text $s 320 230 214 24 'TRANSFORM SET  T' 16 $C.Orange $true 2 | Out-Null
    Add-Text $s 320 263 214 24 'Target · action · direction' 11 $C.Gray $false 2 | Out-Null
    Add-Line $s 554 191 640 191 $C.Gray 2.5 $true | Out-Null
    Add-Box $s 642 128 270 126 $C.Panel $C.Navy2 5 1.5 | Out-Null
    Add-Text $s 660 148 234 26 'Generated video  V' 17 $C.Navy $true 2 | Out-Null
    Add-Text $s 660 187 234 42 'Execution · Preservation · Coherence' 12 $C.Gray $false 2 | Out-Null
    Add-Text $s 42 330 870 24 'Five evaluation dimensions' 15 $C.Navy $true | Out-Null
    $dims = @(
        @('Attribute','Binding',$C.Blue), @('Action','Binding',$C.Cyan), @('Motion','Binding',$C.Orange),
        @('Background','Dynamics',$C.Green), @('View','Transformation',$C.Navy2)
    )
    for ($i=0; $i -lt 5; $i++) {
        $x = 42 + $i*176
        Add-Box $s $x 365 156 72 $C.White $dims[$i][2] 5 1.5 | Out-Null
        Add-Text $s ($x+8) 378 140 20 $dims[$i][0] 13 $dims[$i][2] $true 2 | Out-Null
        Add-Text $s ($x+8) 404 140 18 $dims[$i][1] 10 $C.Gray $false 2 | Out-Null
    }
    Add-Text $s 42 463 870 24 'Valid(x) = Identifiable ∧ Observable ∧ Separable ∧ Non-trivial' 13 $C.Navy2 $true 2 'Cambria Math' | Out-Null

    Write-Output 'Building slide 5...'
    # Slide 5
    $s = New-ContentSlide $presentation 'Overall Pipeline Architecture' 5
    $phases = @(
        @('TIP-I2V','Real user priors',$C.Navy2), @('Phase 1','Prior extraction',$C.Blue),
        @('Phase 2','Candidate synthesis',$C.Cyan), @('Quality Control','P0–P4 repair',$C.Orange),
        @('Validity','Model + human study',$C.Green)
    )
    for ($i=0; $i -lt 5; $i++) {
        $x = 34 + $i*184
        Add-Box $s $x 126 154 92 $C.White $phases[$i][2] 5 1.7 | Out-Null
        Add-Text $s ($x+8) 144 138 24 $phases[$i][0] 15 $phases[$i][2] $true 2 | Out-Null
        Add-Text $s ($x+8) 180 138 20 $phases[$i][1] 10 $C.Gray $false 2 | Out-Null
        if ($i -lt 4) { Add-Line $s ($x+155) 172 ($x+183) 172 $C.Gray 2.2 $true | Out-Null }
    }
    Add-Text $s 36 258 888 24 'Three design principles' 16 $C.Navy $true | Out-Null
    Add-Card $s 36 296 278 120 '01  Prior traceability' 'Semantic recipes remain linked to real user data.' $C.Blue $C.PaleBlue 15 12
    Add-Card $s 341 296 278 120 '02  Dimension isolation' 'One primary intervention per benchmark item.' $C.Cyan $C.PaleCyan 15 12
    Add-Card $s 646 296 278 120 '03  Structured delivery' 'Evaluators consume explicit target fields.' $C.Orange $C.PaleOrange 15 12
    Add-Text $s 36 455 888 30 'Midterm status: candidate generation operational · repair/audit in progress · validity study pending' 11 $C.Red $true 2 | Out-Null

    Write-Output 'Building slide 6...'
    # Slide 6
    $s = New-ContentSlide $presentation 'Phase 1 — Prior Data Preparation' 6
    Add-Text $s 36 76 888 22 'Upper pipeline · model-assisted parsing' 13 $C.Navy $true | Out-Null
    $upper = @('Scan & clean','VLM visual parse','LLM intent parse','Cross-modal check','Prior Package')
    for ($i=0; $i -lt 5; $i++) {
        $x=36+$i*178
        Add-Box $s $x 110 148 58 $C.PaleBlue $C.Blue 5 1.2 | Out-Null
        Add-Text $s ($x+8) 126 132 24 $upper[$i] 11 $C.Navy $true 2 | Out-Null
        if($i -lt 4){Add-Line $s ($x+149) 139 ($x+177) 139 $C.Gray 1.8 $true | Out-Null}
    }
    Add-Text $s 36 198 888 22 'Lower pipeline · deterministic enhancement' 13 $C.Navy $true | Out-Null
    $lower=@('patch','align','refbank','priors2','recipes','audit')
    for($i=0;$i -lt 6;$i++){
        $x=36+$i*148
        Add-Box $s $x 232 120 50 $C.PaleCyan $C.Cyan 5 1.2 | Out-Null
        Add-Text $s ($x+8) 246 104 20 $lower[$i] 11 $C.Cyan $true 2 'Consolas' | Out-Null
        if($i -lt 5){Add-Line $s ($x+121) 257 ($x+147) 257 $C.Gray 1.5 $true | Out-Null}
    }
    Add-Card $s 36 322 280 112 'Reproducibility' "Zero additional API calls`nDeterministic · idempotent" $C.Green $C.PaleGreen 15 12
    Add-Card $s 340 322 280 112 'Instance matching' "Exact = 1.0`nSubstring = 0.7`nUnmatched = 0.0" $C.Blue $C.PaleBlue 15 12
    Add-Card $s 644 322 280 112 'Quality gate' "Low-confidence alignment`n→ review or reject" $C.Orange $C.PaleOrange 15 12
    Add-Text $s 36 462 888 24 'Constraint note: “zero API” applies to enhancement modules, not to upstream VLM/LLM parsing.' 10 $C.Gray $false 2 | Out-Null

    Write-Output 'Building slide 7...'
    # Slide 7
    $s = New-ContentSlide $presentation 'Phase 2 — Dataset Synthesis' 7
    $steps=@('Quota','Sample','Plan','Construct','Verify','Finalize','Export','Audit')
    for($i=0;$i -lt 8;$i++){
        $x=26+$i*116
        Add-Box $s $x 100 92 48 $C.PaleBlue $C.Blue 5 1.1 | Out-Null
        Add-Text $s ($x+4) 113 84 19 ($i+1).ToString('00') 9 $C.Gray $true 2 | Out-Null
        Add-Text $s ($x+4) 128 84 18 $steps[$i] 10 $C.Navy $true 2 | Out-Null
        if($i -lt 7){Add-Line $s ($x+93) 124 ($x+115) 124 $C.Gray 1.4 $true | Out-Null}
    }
    Add-Card $s 34 190 276 122 'Exact integer quotas' 'Largest remainder method keeps the total fixed.' $C.Blue $C.PaleBlue 15 12
    Add-Card $s 342 190 276 122 'Traceable dual-track assets' 'TIP-I2V source + native image + 16:9 companion.' $C.Cyan $C.PaleCyan 15 12
    Add-Card $s 650 190 276 122 'Structured candidate checks' 'VQA check + prompt finalization + rule audit.' $C.Orange $C.PaleOrange 15 12
    Add-Text $s 34 347 892 22 'Contrastive controls' 15 $C.Navy $true | Out-Null
    $controls=@('static_copy','random_motion','global_filter','camera_pan_cheat','subject_swap')
    for($i=0;$i -lt 5;$i++){
        $x=34+$i*178
        Add-Box $s $x 383 158 46 $C.Panel $C.LightGray 5 1 | Out-Null
        Add-Text $s ($x+6) 397 146 18 $controls[$i] 10 $C.Gray $true 2 'Consolas' | Out-Null
    }
    Add-Text $s 34 458 892 30 'Release rule: file generation ≠ schema validity. Structural completeness is a hard gate.' 11 $C.Red $true 2 | Out-Null

    Write-Output 'Building slide 8...'
    # Slide 8
    $s = New-ContentSlide $presentation 'Data Funnel & Current Status' 8
    $funnel=@(
        @(4092,'Question plans',560,$C.Navy2), @(3519,'Prompt candidates',470,$C.Blue),
        @(3517,'Manifest rows',380,$C.Cyan), @(1500,'Release target',290,$C.Orange)
    )
    for($i=0;$i -lt 4;$i++){
        $w=$funnel[$i][2]; $x=52+(560-$w)/2; $y=94+$i*76
        Add-Box $s $x $y $w 55 $C.Panel $funnel[$i][3] 5 1.5 | Out-Null
        Add-Text $s ($x+18) ($y+9) 110 30 ([string]::Format('{0:N0}',$funnel[$i][0])) 21 $funnel[$i][3] $true | Out-Null
        Add-Text $s ($x+136) ($y+15) ($w-150) 24 $funnel[$i][1] 13 $C.Ink $true | Out-Null
    }
    Add-Text $s 52 410 560 28 '85.9% = manifest coverage, not final QC pass rate' 11 $C.Red $true 2 | Out-Null
    Add-Card $s 646 96 278 100 'Image assets' '8,184 = 4,092 native + 4,092 companions' $C.Blue $C.PaleBlue 15 12
    Add-Card $s 646 214 278 100 'Fallback prompts' '117 / 3,519 = 3.3%' $C.Cyan $C.PaleCyan 15 12
    Add-Card $s 646 332 278 112 'Release status' "Target: 5 × 300 samples`nPending final structural audit" $C.Orange $C.PaleOrange 15 12
    Add-Text $s 52 466 872 24 'Candidate assets are not yet a validated benchmark release.' 12 $C.Navy $true 2 | Out-Null

    Write-Output 'Building slide 9...'
    # Slide 9
    $s = New-ContentSlide $presentation 'Quality Issues — P0 Critical Finding' 9
    Add-Text $s 36 76 888 22 'Structured evaluation targets remain the blocking issue.' 14 $C.Red $true | Out-Null
    Add-Card $s 36 112 276 118 'Frozen old baseline' "Subject noun: 0 / 3,517`nStructured change: 0 / 3,517" $C.Red $C.PaleRed 15 12
    Add-Card $s 342 112 276 118 'Current repair snapshot' "Subject noun: 3,263 / 3,517`nCoverage: 92.8%" $C.Green $C.PaleGreen 15 12
    Add-Card $s 648 112 276 118 'Residual blocker' "Generic/empty subjects: 254`ntarget_relation: 0 / 3,517" $C.Orange $C.PaleOrange 15 12
    Add-Text $s 36 262 888 22 'Root-cause path' 15 $C.Navy $true | Out-Null
    $causes=@('Phase 1 fields','Field/version adapter','Phase 2 targets','Export hard gates')
    for($i=0;$i -lt 4;$i++){
        $x=36+$i*222
        $lineColor=$(if($i -eq 1 -or $i -eq 2){$C.Red}else{$C.Gray})
        Add-Box $s $x 300 190 62 $C.Panel $lineColor 5 1.4 | Out-Null
        Add-Text $s ($x+10) 319 170 22 $causes[$i] 12 $lineColor $true 2 | Out-Null
        if($i -lt 3){Add-Line $s ($x+191) 331 ($x+221) 331 $C.Gray 2 $true | Out-Null}
    }
    Add-Line $s 36 405 924 405 $C.Red 3 | Out-Null
    Add-Text $s 42 416 876 30 'P0 partially repaired — manifest and preliminary “Final 1500” are NOT release-ready.' 13 $C.Red $true 2 | Out-Null
    Add-Text $s 42 463 876 24 'Next: reconstruct relations → resolve residuals → full audit → freeze version → package' 11 $C.Navy2 $true 2 | Out-Null

    Write-Output 'Building slide 10...'
    # Slide 10
    $s = New-ContentSlide $presentation 'Quality Issues — P1 to P4' 10
    $issues=@(
        @('P1  Low-frequency wording','5.8% flagged; keep necessary terms',$C.Orange,'Insert prompt example'),
        @('P2  Image clarity','Low-resolution sources; SR may hallucinate',$C.Red,'Insert clarity comparison'),
        @('P3  Aspect ratio','AR 0.53–2.52; crop/pad may alter targets',$C.Red,'Insert AR collage'),
        @('P4  Distribution','Subject frequency and difficulty uncalibrated',$C.Orange,'Insert long-tail chart')
    )
    for($i=0;$i -lt 4;$i++){
        $x=$(if($i%2 -eq 0){36}else{488}); $y=$(if($i -lt 2){88}else{290})
        Add-Box $s $x $y 436 178 $C.White $issues[$i][2] 5 1.4 | Out-Null
        Add-Text $s ($x+16) ($y+12) 404 24 $issues[$i][0] 15 $issues[$i][2] $true | Out-Null
        Add-Placeholder $s ($x+16) ($y+48) 150 102 $issues[$i][3] 'Optional visual' | Out-Null
        Add-Text $s ($x+182) ($y+58) 236 78 $issues[$i][1] 12 $C.Ink $true | Out-Null
    }
    Add-Text $s 36 482 888 24 'Shared rule: quality repair must not change the semantic ground truth.' 12 $C.Navy $true 2 | Out-Null

    Write-Output 'Building slide 11...'
    # Slide 11
    $s = New-ContentSlide $presentation 'Future Work — Repair & Validation' 11
    Add-Card $s 34 86 286 174 '1  P0 closure & versioning' "Reconstruct target_relation`nResolve/reject 254 subjects`nAdd hard gates and freeze manifest" $C.Red $C.PaleRed 15 12
    Add-Card $s 337 86 286 174 '2  Quality strategy comparison' "Meaning-preserving text simplification`nCompare interpolation / sharpening / SR`nRecord aspect-ratio adaptation" $C.Orange $C.PaleOrange 15 12
    Add-Card $s 640 86 286 174 '3  Ablation & validity' "50–100 items per dimension`n2–3 image+text I2V models`nAutomatic metrics + blind human ratings" $C.Green $C.PaleGreen 15 12
    Add-Text $s 34 292 892 22 'Validity loop' 15 $C.Navy $true | Out-Null
    $valid=@('Frozen sample','Model inference','Proxy metrics','Blind ratings','Agreement analysis')
    for($i=0;$i -lt 5;$i++){
        $x=34+$i*178
        Add-Box $s $x 330 150 54 $C.PaleBlue $C.Blue 5 1.1 | Out-Null
        Add-Text $s ($x+8) 347 134 20 $valid[$i] 11 $C.Navy $true 2 | Out-Null
        if($i -lt 4){Add-Line $s ($x+151) 357 ($x+177) 357 $C.Gray 1.8 $true | Out-Null}
    }
    Add-Text $s 34 416 892 42 "Tests: dimension discriminability · difficulty ordering · failure-mode validity`nNo model ranking is assumed in advance." 12 $C.Navy2 $true 2 | Out-Null

    Write-Output 'Building slide 12...'
    # Slide 12
    $s = New-ContentSlide $presentation 'Timeline — 12-Week Plan' 12
    $tasks=@(
        @('Close P0 + freeze',1,2,$C.Red,'Audited manifest'),
        @('Calibrate P1–P4',3,4,$C.Orange,'Frozen quality rules'),
        @('Build release candidate',5,6,$C.Blue,'5 × 300 package'),
        @('Validity + human study',7,9,$C.Cyan,'Agreement statistics'),
        @('Thesis + dataset card',10,12,$C.Green,'Draft + documentation')
    )
    $left=230; $gridW=684; $weekW=$gridW/12
    Add-Text $s 36 88 180 24 'Work package' 13 $C.Navy $true | Out-Null
    for($w=1;$w -le 12;$w++){
        $x=$left+($w-1)*$weekW
        Add-Text $s $x 88 $weekW 24 ("W"+$w) 10 $C.Gray $true 2 | Out-Null
        Add-Line $s $x 116 $x 432 $C.LightGray 0.8 | Out-Null
    }
    Add-Line $s ($left+$gridW) 116 ($left+$gridW) 432 $C.LightGray 0.8 | Out-Null
    for($i=0;$i -lt 5;$i++){
        $y=130+$i*62
        Add-Text $s 36 $y 184 22 $tasks[$i][0] 11 $C.Ink $true | Out-Null
        Add-Text $s 36 ($y+24) 184 18 $tasks[$i][4] 9 $C.Gray | Out-Null
        $x=$left+($tasks[$i][1]-1)*$weekW+3
        $w=($tasks[$i][2]-$tasks[$i][1]+1)*$weekW-6
        Add-Box $s $x ($y+4) $w 28 $tasks[$i][3] $tasks[$i][3] 5 0 | Out-Null
        Add-Text $s $x ($y+8) $w 18 ("W"+$tasks[$i][1]+"–W"+$tasks[$i][2]) 9 $C.White $true 2 | Out-Null
    }
    Add-Text $s 36 462 888 24 'Each stage ends with an auditable deliverable—not merely a completed run.' 11 $C.Navy2 $true 2 | Out-Null

    Write-Output 'Building slide 13...'
    # Slide 13
    $s = $presentation.Slides.Add($presentation.Slides.Count + 1, $ppLayoutBlank)
    $s.FollowMasterBackground = $msoFalse
    $s.Background.Fill.ForeColor.RGB = $C.Navy
    Add-Text $s 44 36 760 42 'Interim Contributions & Next Phase' 26 $C.White $true | Out-Null
    $contrib=@(
        @('01','Preserve–Transform framework','Explicit change + preservation constraints',$C.Cyan),
        @('02','Prior-grounded pipeline','Candidate construction traceable to TIP-I2V',$C.Blue),
        @('03','Candidate assets','3,519 prompts · 8,184 images · target 1,500',$C.Gold),
        @('04','Auditable quality workflow','P0–P4 across schema, text, image, AR, distribution',$C.Orange)
    )
    for($i=0;$i -lt 4;$i++){
        $x=$(if($i%2 -eq 0){44}else{470}); $y=$(if($i -lt 2){112}else{258})
        Add-Box $s $x $y 392 118 $C.Navy2 $contrib[$i][3] 5 1.3 | Out-Null
        Add-Text $s ($x+18) ($y+14) 54 30 $contrib[$i][0] 20 $contrib[$i][3] $true | Out-Null
        Add-Text $s ($x+78) ($y+14) 290 26 $contrib[$i][1] 15 $C.White $true | Out-Null
        Add-Text $s ($x+78) ($y+52) 290 44 $contrib[$i][2] 11 $C.LightGray | Out-Null
    }
    Add-Text $s 44 410 884 26 'Next: close P0 → compare quality strategies → validate metrics → freeze release candidate' 12 $C.Gold $true 2 | Out-Null
    Add-Text $s 44 458 884 34 'Thank You · Questions Welcome' 22 $C.White $true 2 | Out-Null

    $fullOutput = [System.IO.Path]::GetFullPath($OutputPath)
    $outputDir = [System.IO.Path]::GetDirectoryName($fullOutput)
    if (-not (Test-Path -LiteralPath $outputDir)) { New-Item -ItemType Directory -Path $outputDir | Out-Null }
    if (Test-Path -LiteralPath $fullOutput) { Remove-Item -LiteralPath $fullOutput -Force }
    Write-Output 'Saving presentation...'
    $presentation.SaveAs($fullOutput, $ppSaveAsOpenXMLPresentation)
    Write-Output "Saved: $fullOutput"
    Write-Output "Slides: $($presentation.Slides.Count)"
}
catch {
    Write-Error ("PPT generation failed: " + $_.Exception.Message)
    throw
}
finally {
    if ($presentation -ne $null) {
        $presentation.Saved = $msoTrue
        $presentation.Close()
    }
    if ($app -ne $null) { $app.Quit() }
    if ($presentation -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
    if ($app -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
