struct AlzheimerAction
    name::Symbol
end

Base.:(==)(a::AlzheimerAction, b::AlzheimerAction) = a.name == b.name
Base.hash(a::AlzheimerAction, h::UInt) = hash(a.name, h)

const ACTIONS = [AlzheimerAction(:A_Wait), AlzheimerAction(:A_Test_MMSE), AlzheimerAction(:A_Test_CDR),
                 AlzheimerAction(:A_Test_APOE4), AlzheimerAction(:A_Treat_MCI), AlzheimerAction(:A_Treat_Dementia)]

isterminal_action(a::AlzheimerAction) = a.name in (:A_Wait, :A_Treat_MCI, :A_Treat_Dementia)
is_test(a::AlzheimerAction) = a.name in (:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4)
