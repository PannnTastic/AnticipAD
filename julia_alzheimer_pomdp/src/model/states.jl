struct AlzheimerState
    name::Symbol
end

Base.:(==)(a::AlzheimerState, b::AlzheimerState) = a.name == b.name
Base.hash(s::AlzheimerState, h::UInt) = hash(s.name, h)

const STATES = [AlzheimerState(:CN), AlzheimerState(:MCI), AlzheimerState(:Dementia)]
