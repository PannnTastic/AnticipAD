struct AlzheimerObs
    name::Symbol
end

Base.:(==)(a::AlzheimerObs, b::AlzheimerObs) = a.name == b.name
Base.hash(o::AlzheimerObs, h::UInt) = hash(o.name, h)

const OBSERVATIONS = [AlzheimerObs(:None), AlzheimerObs(:MMSE_Normal), AlzheimerObs(:MMSE_Sedang), AlzheimerObs(:MMSE_Rendah),
                      AlzheimerObs(:CDR_0), AlzheimerObs(:CDR_0_5), AlzheimerObs(:CDR_1_plus),
                      AlzheimerObs(:APOE4_Negatif), AlzheimerObs(:APOE4_Hetero), AlzheimerObs(:APOE4_Homo)]
