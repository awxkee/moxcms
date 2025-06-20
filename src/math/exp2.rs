/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::math::common::*;

pub(crate) const TBLSIZE: usize = 256;
#[repr(align(64))]
pub(crate) struct Exp2Table(pub(crate) [(u64, u64); TBLSIZE]);

#[rustfmt::skip]
pub(crate) static EXP2FT: Exp2Table = Exp2Table([(0x3FE6A09E667F3BCD, 0x3C9C6CDCB8CFCC24),(0x3FE6B052FA75173E, 0xBC7AADA2CD342900),(0x3FE6C012750BDABF, 0x3C72CEC326EE0B73),(0x3FE6CFDCDDD47645, 0xBC9CD152FC304554),(0x3FE6DFB23C651A2F, 0x3C6BFF379BAC560E),(0x3FE6EF9298593AE5, 0x3C90D4F348699267),(0x3FE6FF7DF9519484, 0x3C885304766E7062),(0x3FE70F7466F42E87, 0xBC59DCCB590677C3),(0x3FE71F75E8EC5F74, 0x3C81669AA6AEA2D3),(0x3FE72F8286EAD08A, 0x3C91F636E60CA2FB),(0x3FE73F9A48A58174, 0x3C908A8544A7C319),(0x3FE74FBD35D7CBFD, 0xBC901F28F3ED6D87),(0x3FE75FEB564267C9, 0x3C8FE19BB8ECD7C4),(0x3FE77024B1AB6E09, 0xBC9B0DF7B199D826),(0x3FE780694FDE5D3F, 0xBC97F77C42EAF147),(0x3FE790B938AC1CF6, 0xBC92E49EF3FC3A27),(0x3FE7A11473EB0187, 0x3C839EA4B4E0A91D),(0x3FE7B17B0976CFDB, 0x3C9B33A31CEFD699),(0x3FE7C1ED0130C132, 0xBC9E308095C8FAB2),(0x3FE7D26A62FF86F0, 0xBC9130FF6D7E82A3),(0x3FE7E2F336CF4E62, 0xBC6FA03BD08BCDA3),(0x3FE7F3878491C491, 0x3C7FCBFD9B8BCF20),(0x3FE80427543E1A12, 0x3C91C4A6D87E2162),(0x3FE814D2ADD106D9, 0xBC938BD3AB087882),(0x3FE82589994CCE13, 0x3C9C01C85ABDA967),(0x3FE8364C1EB941F7, 0xBC9869EC57356F88),(0x3FE8471A4623C7AD, 0x3C879DA32BFD8963),(0x3FE857F4179F5B21, 0x3C5A38BCDFA97920),(0x3FE868D99B4492ED, 0x3C9E0CE7D31C17B4),(0x3FE879CAD931A436, 0xBC8494F5F00A618D),(0x3FE88AC7D98A6699, 0xBC980F72957B0484),(0x3FE89BD0A478580F, 0xBC9B822CE5C61B11),(0x3FE8ACE5422AA0DB, 0xBC956F6CD720FBA3),(0x3FE8BE05BAD61778, 0xBC9CBABB6DF10102),(0x3FE8CF3216B5448C, 0x3C6F530B22C146B2),(0x3FE8E06A5E0866D9, 0x3C95678B261581ED),(0x3FE8F1AE99157736, 0xBC842BC69235780F),(0x3FE902FED0282C8A, 0xBC93E8F1D3D2ED6D),(0x3FE9145B0B91FFC6, 0x3C9B766E6AA0A1DF),(0x3FE925C353AA2FE2, 0x3C81B06301EF3FBF),(0x3FE93737B0CDC5E5, 0x3C6565B633AAADEE),(0x3FE948B82B5F98E5, 0x3C8B2C8AD636D047),(0x3FE95A44CBC8520F, 0x3C744C91B60B1F8D),(0x3FE96BDD9A7670B3, 0x3C591A99420A1E7A),(0x3FE97D829FDE4E50, 0x3C9A58FA485235A1),(0x3FE98F33E47A22A2, 0xBC79E4BF979F39CA),(0x3FE9A0F170CA07BA, 0x3C8F6FF7608A3078),(0x3FE9B2BB4D53FE0D, 0x3C9ACED3C086ACE9),(0x3FE9C49182A3F090, 0xBC7984806660CAA6),(0x3FE9D674194BB8D5, 0x3C92D72D9EC09AA5),(0x3FE9E86319E32323, 0xBC7582EB35CAC3F3),(0x3FE9FA5E8D07F29E, 0x3C825C551918A750),(0x3FEA0C667B5DE565, 0x3C9125694BB46C7E),(0x3FEA1E7AED8EB8BB, 0xBC99190F3B11997B),(0x3FEA309BEC4A2D33, 0xBC938E88295250EF),(0x3FEA42C980460AD8, 0x3C976DCE520D757A),(0x3FEA5503B23E255D, 0x3C999593455A26B5),(0x3FEA674A8AF46052, 0xBC626959DC7D1A72),(0x3FEA799E1330B358, 0xBC983BDE4707301A),(0x3FEA8BFE53C12E59, 0x3C923BF4E59C1A46),(0x3FEA9E6B5579FDBF, 0xBC8D72D3F1E40A93),(0x3FEAB0E521356EBA, 0xBC854895B32439F7),(0x3FEAC36BBFD3F37A, 0x3C8B3AD1802C5436),(0x3FEAD5FF3A3C2774, 0xBC949662999DD625),(0x3FEAE89F995AD3AD, 0xBC9445B6038D7018),(0x3FEAFB4CE622F2FF, 0x3C91B55866B577A1),(0x3FEB0E07298DB666, 0x3C97C789B4248453),(0x3FEB20CE6C9A8952, 0xBC91C09EECE01858),(0x3FEB33A2B84F15FB, 0x3C5F667A98D32AC1),(0x3FEB468415B749B1, 0x3C7AA046DA7AC268),(0x3FEB59728DE5593A, 0x3C9801F60C88F9BD),(0x3FEB6C6E29F1C52A, 0xBC8F69C45CB956A0),(0x3FEB7F76F2FB5E47, 0x3C71EB0404780AD1),(0x3FEB928CF22749E4, 0x3C96FA19E9BA96C0),(0x3FEBA5B030A1064A, 0x3C99DF3F08EDA153),(0x3FEBB8E0B79A6F1F, 0x3C3A14F11F70B628),(0x3FEBCC1E904BC1D2, 0xBC7E4BB29D39C3A3),(0x3FEBDF69C3F3A207, 0x3C374FD9C736D243),(0x3FEBF2C25BD71E09, 0x3C9998B2BE294A3F),(0x3FEC06286141B33D, 0x3C88550839B2BF30),(0x3FEC199BDD85529C, 0xBC7C08F260A0042A),(0x3FEC2D1CD9FA652C, 0x3C92C1A4A8948B34),(0x3FEC40AB5FFFD07A, 0xBC9647D6355E9B74),(0x3FEC544778FAFB22, 0xBC8C00CDBAF8D243),(0x3FEC67F12E57D14B, 0xBC8E1E92A1C04F0D),(0x3FEC7BA88988C933, 0x3C88B038839477F4),(0x3FEC8F6D9406E7B5, 0xBC6C920234DBF34D),(0x3FECA3405751C4DB, 0x3C834D9FA0DD4CB4),(0x3FECB720DCEF9069, 0xBC70E49A8304EF1F),(0x3FECCB0F2E6D1675, 0x3C775B06D4E67421),(0x3FECDF0B555DC3FA, 0x3C87DC824B04C033),(0x3FECF3155B5BAB74, 0x3C94C2508F696AFD),(0x3FED072D4A07897C, 0x3C96D9AB77FF22EF),(0x3FED1B532B08C968, 0xBC90EBCF9B6D36B2),(0x3FED2F87080D89F2, 0x3C972904080A5C48),(0x3FED43C8EACAA1D6, 0xBC8F53213EADD172),(0x3FED5818DCFBA487, 0xBC7DC682D68358E2),(0x3FED6C76E862E6D3, 0xBC59083C825660B4),(0x3FED80E316C98398, 0x3C8ACA05497DE9C9),(0x3FED955D71FF6075, 0xBC944D86AE32150E),(0x3FEDA9E603DB3285, 0xBC95E51917091778),(0x3FEDBE7CD63A8315, 0x3C95506A5083B5B0),(0x3FEDD321F301B460, 0xBC8D2EC7BFC90FD9),(0x3FEDE7D5641C0658, 0x3C961C57EFD0005C),(0x3FEDFC97337B9B5F, 0x3C8B2B6B3F445F87),(0x3FEE11676B197D17, 0x3C6CB94223DCB497),(0x3FEE264614F5A129, 0x3C92277152860520),(0x3FEE3B333B16EE12, 0x3C93D186DB10A7E8),(0x3FEE502EE78B3FF6, 0xBC7DE1319ABB2AE2),(0x3FEE653924676D76, 0x3C80E5ADF9AD21B7),(0x3FEE7A51FBC74C83, 0xBC8C86CFB6CD46E4),(0x3FEE8F7977CDB740, 0x3C89BB5327E99E88),(0x3FEEA4AFA2A490DA, 0x3C970ED6CCCEEC4B),(0x3FEEB9F4867CCA6E, 0xBC8ED1E0A526B1E2),(0x3FEECF482D8E67F1, 0x3C95693AB0188765),(0x3FEEE4AAA2188510, 0xBC8A90422E519191),(0x3FEEFA1BEE615A27, 0xBC963124664AAF80),(0x3FEF0F9C1CB6412A, 0x3C8C6FF2F42E7105),(0x3FEF252B376BBA97, 0xBC8D19790D732AB4),(0x3FEF3AC948DD7274, 0x3C72BD4A13B9B3AB),(0x3FEF50765B6E4540, 0xBC9309E68DA0499F),(0x3FEF6632798844F8, 0xBC974254F6C56244),(0x3FEF7BFDAD9CBE14, 0x3C95CC2651E079FB),(0x3FEF91D802243C89, 0x3C59206B311D2332),(0x3FEFA7C1819E90D8, 0xBC80FA4931906B80),(0x3FEFBDBA3692D514, 0x3C7279822F498FD4),(0x3FEFD3C22B8F71F1, 0xBC5B71923D729650),(0x3FEFE9D96B2A23D9, 0xBC6DDEC82C0249D6),(0x3FF0000000000000, 0x00000000),(0x3FF00B1AFA5ABCBF, 0x3C8E2992B5FD7D1B),(0x3FF0163DA9FB3335, 0xBCA3A4BC88AB6F0B),(0x3FF02168143B0281, 0x3C8AD3C1DBD6C7CA),(0x3FF02C9A3E778061, 0x3C791137D36A5DAC),(0x3FF037D42E11BBCC, 0xBC6E77C22DC99C3F),(0x3FF04315E86E7F85, 0x3C979D9482B6E550),(0x3FF04E5F72F654B1, 0xBC8D64A18DD44CAD),(0x3FF059B0D3158574, 0xBC94CA556A1B2FA8),(0x3FF0650A0E3C1F89, 0x3C9EAFB52FB794FC),(0x3FF0706B29DDF6DE, 0x3C940EE9A808DE1D),(0x3FF07BD42B72A836, 0xBC8ACCA634F5A05F),(0x3FF0874518759BC8, 0xBC687A10F84F53B7),(0x3FF092BDF66607E0, 0x3C9F570D8081907E),(0x3FF09E3ECAC6F383, 0xBC9801B99F4F5B18),(0x3FF0A9C79B1F3919, 0xBC8E39596D3F89AF),(0x3FF0B5586CF9890F, 0xBCA106D9BA253401),(0x3FF0C0F145E46C85, 0xBC9CE61CF1EF2524),(0x3FF0CC922B7247F7, 0xBC96269675FF4E35),(0x3FF0D83B23395DEC, 0x3CA304633DFF3098),(0x3FF0E3EC32D3D1A2, 0xBC462D2D983E6844),(0x3FF0EFA55FDFA9C5, 0x3C9C1949D4C17CE2),(0x3FF0FB66AFFED31B, 0x3C72C399E6B423B5),(0x3FF1073028D7233E, 0xBC93D80633BDBA39),(0x3FF11301D0125B51, 0x3C9EC8769E7386F7),(0x3FF11EDBAB5E2AB6, 0x3CA34EFBE2272E3E),(0x3FF12ABDC06C31CC, 0x3C57CF4B389B60D3),(0x3FF136A814F204AB, 0x3C6EEDF4012C3DEA),(0x3FF1429AAEA92DE0, 0x3C99A8CAB70C1007),(0x3FF14E95934F312E, 0x3C9262BB46B2B386),(0x3FF15A98C8A58E51, 0xBC8846F8D141B1D0),(0x3FF166A45471C3C2, 0xBC608BCF38881F6D),(0x3FF172B83C7D517B, 0x3C873C5AEC1A0547),(0x3FF17ED48695BBC0, 0xBC75ECF8C1A3E48C),(0x3FF18AF9388C8DEA, 0x3C9673A724B46D55),(0x3FF1972658375D2F, 0xBC9B1EF6533A7736),(0x3FF1A35BEB6FCB75, 0xBC93DD2C819E0908),(0x3FF1AF99F8138A1C, 0xBC9EFECBA1D39F6E),(0x3FF1BBE084045CD4, 0x3CA07B9A3E9B6A86),(0x3FF1C82F95281C6B, 0xBC94D16266C5F6A1),(0x3FF1D4873168B9AA, 0xBCA36C3949DFC34E),(0x3FF1E0E75EB44027, 0x3C9DAF432E2DB466),(0x3FF1ED5022FCD91D, 0x3C97039D1EA652DC),(0x3FF1F9C18438CE4D, 0x3CA1F35F0F6AA245),(0x3FF2063B88628CD6, 0xBC93118AE698D206),(0x3FF212BE3578A819, 0xBC98B6332CA946BB),(0x3FF21F49917DDC96, 0xBC87C551DB68456F),(0x3FF22BDDA27912D1, 0xBC928D041D5EFFB1),(0x3FF2387A6E756238, 0xBCA045C6A6D3EECB),(0x3FF2451FFB82140A, 0xBC90F00211EFDCCD),(0x3FF251CE4FB2A63F, 0xBC90DB2BD14ACF32),(0x3FF25E85711ECE75, 0xBC98FBBB6F35251E),(0x3FF26B4565E27CDD, 0xBC877BF4B6A48214),(0x3FF2780E341DDF29, 0xBCA2C3690668D33F),(0x3FF284DFE1F56381, 0x3CA063B457A17EDB),(0x3FF291BA7591BB70, 0x3C875E450B18C4FD),(0x3FF29E9DF51FDEE1, 0xBC8B5D8E0885AB1B),(0x3FF2AB8A66D10F13, 0x3C9F54A762D37BEE),(0x3FF2B87FD0DAD990, 0x3C45038687737BEE),(0x3FF2C57E39771B2F, 0x3C99D46429F6679F),(0x3FF2D285A6E4030B, 0xBC93A1FEE8C9225C),(0x3FF2DF961F641589, 0xBCA1C9EADEC7A28B),(0x3FF2ECAFA93E2F56, 0xBC75B2C9EFF60911),(0x3FF2F9D24ABD886B, 0x3C69D501EA1A953B),(0x3FF306FE0A31B715, 0xBC8BD9046B69EA24),(0x3FF31432EDEEB2FD, 0xBC8EABAA4F953871),(0x3FF32170FC4CD831, 0xBC900E3E54FB7D75),(0x3FF32EB83BA8EA32, 0x3CA102D77008E4BA),(0x3FF33C08B26416FF, 0xBC96FC3180AAC683),(0x3FF3496266E3FA2D, 0x3C8729B701B77A75),(0x3FF356C55F929FF1, 0x3C905491F94CC017),(0x3FF36431A2DE883B, 0x3C90C7A5A61BFCD0),(0x3FF371A7373AA9CB, 0x3C9A6409CDF25D3A),(0x3FF37F26231E754A, 0x3C9EBC4BA9F5A0F6),(0x3FF38CAE6D05D866, 0x3CA20E5F6745F2E3),(0x3FF39A401B7140EF, 0x3C9E3823AC01DA40),(0x3FF3A7DB34E59FF7, 0x3C79B570DEEB4717),(0x3FF3B57FBFEC6CF4, 0xBC98F1D7BEF807FE),(0x3FF3C32DC313A8E5, 0x3CA21ABAF1248342),(0x3FF3D0E544EDE173, 0xBC8295D0CDB79672),(0x3FF3DEA64C123422, 0xBC8F31B9FB32E2E4),(0x3FF3EC70DF1C5175, 0x3C8F3CFBA6795A4F),(0x3FF3FA4504AC801C, 0x3C9B83C7E85E695F),(0x3FF40822C367A024, 0xBC900E53A5929351),(0x3FF4160A21F72E2A, 0x3C61C8D3F3908470),(0x3FF423FB2709468A, 0x3C9BD214C0D570B9),(0x3FF431F5D950A897, 0x3C8452CBD4449600),(0x3FF43FFA3F84B9D4, 0xBC8BEE60A77E75FB),(0x3FF44E086061892D, 0xBC4BF96EE8AFB9FB),(0x3FF45C2042A7D232, 0x3C6BA748FEF36803),(0x3FF46A41ED1D0057, 0xBCA0283B91B051A3),(0x3FF4786D668B3237, 0x3C9FB804D9A0B540),(0x3FF486A2B5C13CD0, 0xBC7637BD8150DB7D),(0x3FF494E1E192AED2, 0x3C86176C1FBB0F52),(0x3FF4A32AF0D7D3DE, 0xBC9CD9F521D33A36),(0x3FF4B17DEA6DB7D7, 0x3C8320AC36829BBB),(0x3FF4BFDAD5362A27, 0xBC804715F70DA237),(0x3FF4CE41B817C114, 0xBC9228D3276E6D7C),(0x3FF4DCB299FDDD0D, 0xBC9B942D586961A7),(0x3FF4EB2D81D8ABFF, 0x3C97559FA3523937),(0x3FF4F9B2769D2CA7, 0x3C96C785171C0DAA),(0x3FF508417F4531EE, 0xBC7CB1353C04F90D),(0x3FF516DAA2CF6642, 0x3C913802B6C9135A),(0x3FF5257DE83F4EEF, 0x3C7F3815FC4931B8),(0x3FF5342B569D4F82, 0x3C81F0A88738EA6A),(0x3FF542E2F4F6AD27, 0xBC899781B04B3642),(0x3FF551A4CA5D920F, 0x3C8FD7A06DB1D675),(0x3FF56070DDE910D2, 0x3C925676C5F1196E),(0x3FF56F4736B527DA, 0xBC9BB5C29F573DD8),(0x3FF57E27DBE2C4CF, 0x3C91F65AF76D2F4C),(0x3FF58D12D497C7FD, 0xBC83E81D7478CC39),(0x3FF59C0827FF07CC, 0x3C9983CBED438F86),(0x3FF5AB07DD485429, 0xBC97A560F05EEDDB),(0x3FF5BA11FBA87A03, 0x3C9D2E9335450A8F),(0x3FF5C9268A5946B7, 0xBC3DFA723AE1BF17),(0x3FF5D84590998B93, 0x3C9E792723F1B261),(0x3FF5E76F15AD2148, 0xBC9D2407CE0F6433),(0x3FF5F6A320DCEB71, 0x3C8B3D1CA01F0C90),(0x3FF605E1B976DC09, 0x3C94D74BBCD33652),(0x3FF6152AE6CDF6F4, 0xBC9FAA964F163EE5),(0x3FF6247EB03A5585, 0x3C9457F6A88AB15F),(0x3FF633DD1D1929FD, 0xBC993D8C21AE3DEE),(0x3FF6434634CCC320, 0x3C8D5305865039DF),(0x3FF652B9FEBC8FB7, 0x3C9BCE3049ABD5D0),(0x3FF6623882552225, 0x3C9C93B642E08A78),(0x3FF671C1C70833F6, 0x3C8F659D313D064D),(0x3FF68155D44CA973, 0xBC60A350F225390E),(0x3FF690F4B19E9538, 0xBC8891945D585BA8)]);

#[inline]
pub(crate) fn ldexp(d: f64, i: u64) -> f64 {
    let b = d.to_bits();
    f64::from_bits(b.wrapping_add(i.wrapping_shl(52)))
}

/// Computes exp2
///
/// Max found ULP 0.5009765625 with FMA, without FMA 0.50244140625.
///
/// This method based on Gal.
#[inline]
pub fn f_exp2(d: f64) -> f64 {
    const REDUX: f64 = f64::from_bits(0x4338000000000000) / TBLSIZE as f64;

    let ui = f64::to_bits(d);
    let ix = (ui >> 32) & 0x7fffffff;
    if ix >= 0x408ff000 {
        /* |x| >= 1022 or nan */
        if ix >= 0x40900000 && ui >> 63 == 0 {
            /* x >= 1024 or nan */
            /* overflow */
            return d * f64::from_bits(0x4330000000000000);
        }
        if ix >= 0x7ff00000 {
            /* -inf or -nan */
            return -1.0 / d;
        }
        if ui >> 63 != 0 {
            /* x <= -1022 */
            /* underflow */
            if d <= -1075.0 {
                return 0.0;
            }
        }
    } else if ix < 0x3c900000 {
        /* |x| < 5.55112e-17 */
        return 1.0 + d;
    }

    let ui = f64::to_bits(d + REDUX);
    let mut i0 = ui;
    i0 = i0.wrapping_add(TBLSIZE as u64 / 2);
    let k = i0 / TBLSIZE as u64;
    i0 &= TBLSIZE as u64 - 1;
    let mut uf = f64::from_bits(ui);
    uf -= REDUX;

    let stored = EXP2FT.0[i0 as usize];

    let z0: f64 = f64::from_bits(stored.0);
    let z1: f64 = f64::from_bits(stored.1);

    let f: f64 = d - uf - z1;

    let f2 = f * f;
    let w0 = f_fmla(
        f,
        f64::from_bits(0x3f55d7e09b4e3a84),
        f64::from_bits(0x3f83b2abd24650cc),
    );
    let w1 = f_fmla(
        f,
        f64::from_bits(0x3fac6b08d70cf4b5),
        f64::from_bits(0x3fcebfbdff82c424),
    );
    let u = f_fmla(
        f,
        f64::from_bits(0x3fe62e42fefa39ef),
        f_fmla(f2 * f2, w0, f2 * w1),
    );
    ldexp(f_fmla(u, z0, z0), k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp2d() {
        assert_eq!(f_exp2(2.0), 4.0);
        assert_eq!(f_exp2(3.0), 8.0);
        assert_eq!(f_exp2(4.0), 16.0);
        assert!((f_exp2(0.35f64) - 0.35f64.exp2()).abs() < 1e-8);
        assert!((f_exp2(-0.6f64) - (-0.6f64).exp2()).abs() < 1e-8);
    }
}
